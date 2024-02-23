import argparse
import datasets
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from tqdm import tqdm
import csv
import openai
from openai import AzureOpenAI
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore
import json
from datetime import datetime
import tiktoken
import threading


def load_and_split(args):
    print("Loading data from {}".format(args.data_file))
    data = datasets.load_dataset("json", data_files=args.data_file, split="train")
    documents = [Document(text=row) for row in tqdm(data[: args.n_samples]["text"])]

    splitter = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    nodes = splitter.get_nodes_from_documents(documents, show_progress=True)
    passages = [n.text for n in nodes]
    return passages


semaphore = Semaphore(5)
global token_count
token_count = 0
global cum_tokens_count
cum_tokens_count = 0
token_count_lock = threading.Lock()


def count_tokens(text):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5")
    n_tokens = len(tokens)

    with token_count_lock:
        token_count += tokens
        if token_count >= 10000:
            cum_tokens_count += token_count
            print(f"Total tokens processed: {cum_tokens_count}")
            token_count = 0


def parse_gpt_response(response):
    lines = response.split("\n")
    cleaned_lines = []
    for line in lines:
        cleaned_line = "".join(filter(lambda x: x.isalpha() or x.isspace(), line))
        cleaned_lines.append(cleaned_line.strip())
    return cleaned_lines


def generate_queries(passage, client, args):
    positive_prompt = """
        Come up with a list of five possible descriptions that a user might give when linking to the following passage. They should be indirect, and require context to understand -- for example, for a passage discussing a startup that sells GPUs in Fremont, a good description might be "good place to work bay area",  and a bad one would be "startup selling GPU".

        They should not just repeat words or descriptions in the passage -- they should require understanding and context to explain why it's related to the passage. Try to avoid using specific keywords from the passage when more general qualifiers would work equally well. Do not match the tone of the passage -- match a disinterested user looking for/describing what the passage refers to.

        Respond with a list of queries only, and no other extraneous formatting or explanation.
        
        Passage: {}
    # """.format(
        passage
    )

    # negative_prompt = """
    # Given the following passage, come up with 5 varied queries that **seem** to be answered by the document -- they might share related words, or have similar structure, but are in fact NOT answered by information in the document,
    # or necessarily related at all.

    # Respond with a list of queries only, and no other extraneous formatting or explanation.

    # Passage: {}
    # """.format(passage)

    with semaphore:
        # Ensure that we are not sending more than RATE_LIMIT requests per second
        time.sleep(1.0 / args.rate_limit)
        try:
            count_tokens(positive_prompt)
            completion = client.chat.completions.create(
                model="find-new-gpt",
                messages=[
                    {"role": "user", "content": positive_prompt},
                ],
            )
            response_text = completion.choices[0].message.content
            count_tokens(response_text)
            return {
                "passage": passage,
                "queries": parse_gpt_response(response_text),
            }
        except Exception as e:
            print(e, passage)
            return None


def generate_queries_batch(passages, client, args):
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        results = list(
            tqdm(
                executor.map(lambda x: generate_queries(x, client, args), passages),
                total=args.n_samples,
            )
        )
    with open(args.output_file, "a") as f:
        for entry in results:
            if entry is not None:
                json_record = json.dumps(entry)
                f.write(f"{json_record}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--data_file",
        type=str,
        default="./data/c4-train.00000-of-01024.json",
        help="Path to the data file to process",
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
        default="gpt_generated_{}.jsonl".format(
            datetime.now().strftime("%Y%m%d_%H%M%S")
        ),
        help="File to save generated queries",
    )

    args = parser.parse_args()

    client = AzureOpenAI(
        api_key=args.openai_key,
        api_version="2023-05-15",
        base_url="https://find-new-gpt.openai.azure.com/",
    )
    print("Loading data from {}".format(args.data_file))
    passages = load_and_split(args)
    print(f"Loaded {len(passages)} passages")

    print("Generating queries")
    queries = generate_queries_batch(passages, client, args)
    print(f"Generated {len(queries)} queries")


if __name__ == "__main__":
    main()
