import argparse
import datasets
from tqdm import tqdm
import csv
import openai
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
import json
from datetime import datetime
import tiktoken
import threading


def load_local_dataset(args):
    print("Loading data from {}".format(args.data_file))
    passages = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            passages.append(data)
    return passages[: args.n_samples]


def load_iterable_dataset(args):
    print("Loading iterable dataset from huggingface/msmarco")
    dataset = datasets.load_dataset("ms_marco", "v2.1", split="train", streaming=True)
    samples = []
    for i, sample in enumerate(dataset):
        if i >= args.n_samples:
            break
        output = {
            "query": sample["query"],
            "pos": sample["answers"][0],
        }
        samples.append(output)
    return samples


semaphore = Semaphore(50)
token_count = 0
cum_tokens_count = 0
token_count_lock = threading.Lock()


def count_tokens(text, model):
    global token_count
    global cum_tokens_count

    tokenizer = tiktoken.encoding_for_model("gpt-3.5")
    tokens = tokenizer.encode(text)
    n_tokens = len(tokens)

    with token_count_lock:
        token_count += n_tokens
        if token_count >= 30000:
            cum_tokens_count += token_count
            if model == "gpt-35-turbo":
                cost = token_count / 1000 * 0.001
            else:
                cost = token_count / 1000 * 0.02
            print(f"Total tokens processed: {cum_tokens_count}, cost = ${cost} USD")
            token_count = 0


def parse_gpt_response(response):
    lines = response.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_line = "".join(filter(lambda x: x.isalpha() or x.isspace(), line))
        cleaned_lines.append(cleaned_line.strip())
    return cleaned_lines


file_lock = threading.Lock()


def generate_passage(query, passage, args):
    negative_prompt = """
        Here is a text query, and a passage that contains the answer to the query. Your goal is to produce 5 queries that *looks* similar to the original query, but evaluated in formal logic is asking a very different question -- one that is not answered by the passage. Make it so the information needed to answer the new query is not present in the passage.

        One approach is to add a modifier to the query, one that the passage does not answer. For example, if the original query is "What is the average temperature in the Sahara desert?", you could add a negation to the query, like "What is the average temperature in the Sahara desert, when it is not summer?". Make sure that the new query sounds plausible, and asks a question that someone might actually ask.

        Ensure that if you negate the query, the passage does not implicitly answer the negated query. For example, if the original query is "what is a prime number", and you negate it to "what is not a prime number", the passage still implicitly answers the negated query.
        
        Output a numbered list of 5 queries, with no prefix. Do not say "the new query is" or anything like that, or include the original query or passage or reasoning in the response.

        For example:
        1. What is the average temperature in the Sahara desert, when it is not summer?
        2. What is the average temperature in the Gobi desert
        3. When is it hottest in the Sahara desert?
        4. What is the average ground temperature in the Sahara desert?
        5. What is the average body temperature in the Sahara desert?


        Query: {}
        Passage: {}
    """.format(
        query, passage
    )

    positive_prompt = """
        Here is a text query, and a passage that contains the answer to the query. Your goal is to add a qualifier to the query, that refines the question, and makes it more specific. The passage should still contain the answer to the new query. 


        Only respond with a single query, with no prefix. Do not say "the new query is" or anything like that, or include the original query or passage or reasoning in the response.


        Query: {}
        Passage: {}
        """.format(
        query, passage
    )
    with semaphore:
        # Ensure that we are not sending more than RATE_LIMIT requests per second
        time.sleep(50.0 / args.rate_limit)
        try:
            pos_completion = openai.ChatCompletion.create(
                engine="mfine",
                model=args.model,
                messages=[
                    {"role": "user", "content": positive_prompt},
                ],
            )
            pos_response_text = pos_completion.choices[0].message.content
            count_tokens(pos_response_text, args.model)

            neg_completion = openai.ChatCompletion.create(
                engine="mfine",
                model=args.model,
                messages=[
                    {"role": "user", "content": negative_prompt},
                ],
            )
            neg_response_text = neg_completion.choices[0].message.content
            count_tokens(neg_response_text, args.model)

            result = {
                "passage": passage,
                "old_query": pos_response_text,
                "new_queries": parse_gpt_response(neg_response_text),
            }

            if args.verbose:
                print(result)

            # Write result to file
            with file_lock:
                with open(args.output_file, "a") as f:
                    json_record = json.dumps(result)
                    f.write(f"{json_record}\n")

        except Exception as e:
            print(e, passage)


def generate_passages_batch(passages, args):
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        list(
            tqdm(
                executor.map(
                    lambda row: generate_passage(row["query"], row["pos"], args),
                    passages,
                ),
                total=args.n_samples,
            )
        )


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--data_file",
        type=str,
        default="./data/c4-train.00000-of-01024.json",
        help="Path to the data file to process",
    )
    parser.add_argument(
        "--load_from_hf",
        action="store_true",
        default=False,
        help="Load dataset from huggingface/msmarco",
    )

    parser.add_argument(
        "--n_samples", type=int, default=10000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=200,
        help="Size of text chunks to process in each sample",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=5,
        help="Size of overlap between consecutive text chunks",
    )
    parser.add_argument(
        "--include_negative",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-35-turbo",
        help="OpenAI model to use for generation",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--openai_key", type=str, help="OpenAI API key for GPT-3 access"
    )
    parser.add_argument(
        "--max_threads", type=int, default=5, help="Maximum number of threads to use"
    )
    parser.add_argument(
        "--rate_limit",
        type=int,
        default=5,
        help="Rate limit for API requests per second",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="out/gpt_generated_{}.jsonl".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
        help="File to save generated queries",
    )

    args = parser.parse_args()

    openai.api_key = args.openai_key
    openai.api_base = "https://find-new-gpt.openai.azure.com/"
    openai.api_version = "2023-05-15"
    openai.api_type = "azure"

    if os.path.exists(args.output_file):
        print(f"Clearing existing output file: {args.output_file}")
        os.remove(args.output_file)
    if args.load_from_hf:
        passages = load_iterable_dataset(args)
    else:
        print("Loading data from {}".format(args.data_file))
        passages = load_local_dataset(args)
    print(f"Loaded {len(passages)} passages")

    print("Generating queries")
    generate_passages_batch(passages, args)
    print(f"Total tokens processed: {cum_tokens_count+token_count}")


if __name__ == "__main__":
    main()
