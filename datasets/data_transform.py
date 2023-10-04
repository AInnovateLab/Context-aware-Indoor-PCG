import argparse
import logging
from multiprocessing.pool import ThreadPool
from os import getenv
from threading import Lock
from typing import Optional, Tuple

import numpy as np
import openai
import pandas as pd
import pyrate_limiter as pl
from func_timeout import FunctionTimedOut, func_timeout
from tqdm import tqdm

MAX_RETRY = 3
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

count_lock = Lock()
input_tokens_total = 0
output_tokens_total = 0

parser = argparse.ArgumentParser(description="Transform nr3d/sr3d data to generative form.")
parser.add_argument("-i", "--input", type=str, required=True, help="Input file nr3d/sr3d path.")
parser.add_argument("-o", "--output", type=str, required=True, help="Output file path.")
parser.add_argument(
    "-t",
    "--api-token",
    type=str,
    default=None,
    help="OpenAI API token. If not set, will be read from OPENAI_API_KEY env variable.",
)
parser.add_argument("-j", "--threads", type=int, default=12, help="Number of threads to use.")
parser.add_argument("--timeout", type=float, default=30, help="Timeout for OpenAI API requests.")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument(
    "--gpt-engine",
    type=str,
    default="gpt-3.5-turbo",
    choices=["gpt-3.5-turbo", "gpt-4"],
    help="GPT engine to use.",
)
parser.add_argument("--correct-mode", action="store_true", help="Correct the generated sentences.")

limiter = pl.Limiter(pl.RequestRate(3200, pl.Duration.MINUTE))
failed_cases = list()

CANDIDATE_VERB_PROB_DICT = {
    "add": 1.0,
    "put": 1.0,
    "place": 1.0,
    "set": 1.0,
    "create": 1.0,
    "generate": 1.0,
    "insert": 1.0,
    "produce": 1.0,
    "lay": 0.5,
    "deposit": 0.5,
    "position": 0.5,
    "situate": 0.5,
}
CANDIDATE_VERBS = list(CANDIDATE_VERB_PROB_DICT.keys())
CANDIDATE_VERB_PROBS = np.array(list(CANDIDATE_VERB_PROB_DICT.values()))
CANDIDATE_VERB_PROBS = CANDIDATE_VERB_PROBS / CANDIDATE_VERB_PROBS.sum()

# GPT Pricing
USD_PER_GPT35_INPUT_TOKEN = 0.0015 / 1000
USD_PER_GPT35_OUTPUT_TOKEN = 0.0020 / 1000
USD_PER_GPT4_INPUT_TOKEN = 0.03 / 1000
USD_PER_GPT4_INPUT_TOKEN = 0.06 / 1000


def gen_gpt35_prompts():
    return [
        {"role": "system", "content": "You are a helpful chat bot."},
        {
            "role": "user",
            "content": " ".join(
                [
                    "Following sentences locate ONLY ONE object in a scene.",
                    "Transform the sentence to create this object.",
                    "Include generative verbs such as '{}' to create it.".format(
                        np.random.choice(CANDIDATE_VERBS, p=CANDIDATE_VERB_PROBS)
                    ),
                    "Change 'the' to 'a' or 'an' properly.",
                    "Imperative sentences are prefered." if np.random.random() < 0.5 else "",
                    "Declarative sentences such as 'there is' are disallowed.",
                    "Avoid multiple imperative sentences.",
                ]
            ),
        },
    ]


def gen_gpt4_prompts():
    return [
        {"role": "system", "content": "You are a helpful chat bot."},
        {
            "role": "user",
            "content": " ".join(
                [
                    "Following sentences locate ONLY ONE object in a scene.",
                    "Transform the sentence to create this object.",
                    "Include generative verbs such as '{}' to create it.".format(
                        np.random.choice(CANDIDATE_VERBS, p=CANDIDATE_VERB_PROBS)
                    ),
                    "Change 'the' to 'a' or 'an' properly.",
                    # "Imperative sentences are prefered." if np.random.random() < 0.5 else "",
                    "Declarative sentences such as 'there is' are disallowed.",
                    "Avoid multiple imperative sentences.",
                ]
            ),
        },
    ]


def _gpt_transform(gpt_engine: str, raw_utter: str) -> Optional[str]:
    prompts = gen_gpt4_prompts() if gpt_engine == "gpt-4" else gen_gpt35_prompts()
    res = openai.ChatCompletion.create(
        model=gpt_engine,
        temperature=1.25,
        n=1,
        frequency_penalty=0.5,
        messages=prompts + [{"role": "user", "content": raw_utter}],
    )
    text = res.choices[0].message.content
    text = text.strip().split("\n")[0]
    return text, res.usage.prompt_tokens, res.usage.completion_tokens


@limiter.ratelimit("chat_api", delay=True)
def gpt_transform(args) -> Tuple[int, str, Optional[str]]:
    idx, raw_utter, _, timeout, gpt_engine = args
    ret_utter = None
    retry = 0
    while retry < MAX_RETRY:
        try:
            ret_utter, input_tokens_cnt, output_tokens_cnt = func_timeout(
                timeout,
                _gpt_transform,
                args=(
                    gpt_engine,
                    raw_utter,
                ),
            )
            with count_lock:
                global input_tokens_total, output_tokens_total
                input_tokens_total += input_tokens_cnt
                output_tokens_total += output_tokens_cnt
            break
        except FunctionTimedOut:
            retry += 1
            logging.warning(f"Timeout on index {idx}. Retry {retry}/{MAX_RETRY}.")
        except Exception as e:
            retry += 1
            logging.error(f"Exception on index {idx}: {e}. Retry {retry}/{MAX_RETRY}.")
    else:
        logging.error(f"Failed on index {idx}.")
        global failed_cases
        failed_cases.append(idx)

    return idx, raw_utter, ret_utter


def main():
    args = parser.parse_args()

    # set openai api key
    logging.info("Setting OpenAI API key...")
    openai.api_key = args.api_token or getenv("OPENAI_API_KEY")
    assert openai.api_key is not None, "OpenAI API key is not set."
    gpt_engine = args.gpt_engine
    logging.info(f"Using GPT engine {gpt_engine}.")
    # set seed
    logging.info(f"Setting random seed to {args.seed}...")
    np.random.seed(args.seed)

    # read input file
    logging.info(f"Reading input file {args.input}...")
    df = pd.read_csv(args.input, index_col=0 if args.correct_mode else None)
    raw_utters = df["utterance"].tolist()
    insertive_utters = [None] * len(raw_utters)
    if "utterance_generative" in df.columns and not args.correct_mode:
        logging.warning(
            "Column 'utterance_generative' already exists. Existing entries would be ignored."
        )
        insertive_utters = df["utterance_generative"].tolist()
    input_pairs = zip(range(len(raw_utters)), raw_utters, insertive_utters)
    input_pairs = filter(lambda pair: pair[2] is None, input_pairs)
    # add fixed args to input pairs
    input_pairs: Tuple[int, str, Optional[str], float] = list(
        pair + (args.timeout, gpt_engine) for pair in input_pairs
    )
    logging.info(f"Transforming {len(input_pairs)}/{len(raw_utters)} utterances...")

    with ThreadPool(processes=args.threads) as pool:
        output_pairs: Tuple[int, str, Optional[str]] = list(
            tqdm(pool.imap_unordered(gpt_transform, input_pairs), total=len(input_pairs))
        )

    final_pairs = zip(range(len(raw_utters)), raw_utters, insertive_utters)
    final_pairs = list(filter(lambda pair: pair[2] is not None, final_pairs))
    final_pairs.extend(output_pairs)
    final_pairs.sort(key=lambda pair: pair[0])
    assert [pair[0] for pair in final_pairs] == list(range(len(raw_utters)))

    logging.info(f"Failed cases: {failed_cases}.")
    logging.info(f"Used input tokens: {input_tokens_total}.")
    logging.info(f"Used output tokens: {output_tokens_total}.")
    if gpt_engine == "gpt-3.5-turbo":
        logging.info(
            f"Estimated input cost: {input_tokens_total * USD_PER_GPT35_INPUT_TOKEN:.4f} USD."
        )
        logging.info(
            f"Estimated output cost: {output_tokens_total * USD_PER_GPT35_OUTPUT_TOKEN:.4f} USD."
        )
    elif gpt_engine == "gpt-4":
        logging.info(
            f"Estimated input cost: {input_tokens_total * USD_PER_GPT4_INPUT_TOKEN:.4f} USD."
        )
        logging.info(
            f"Estimated output cost: {output_tokens_total * USD_PER_GPT4_INPUT_TOKEN:.4f} USD."
        )
    else:
        logging.error(f"Unknown GPT engine {gpt_engine}.")

    # write output file
    df["utterance_generative"] = [pair[2] for pair in final_pairs]
    df.to_csv(args.output, index=args.correct_mode)
    logging.info(f"Saved to {args.output}.")
    logging.info("Done.")


if __name__ == "__main__":
    main()
