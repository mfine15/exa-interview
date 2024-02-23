python3 src/generate_data.py --n_samples 10 \
--openai_key $OPENAI_API_KEY \
--load_from_hf \
--model gpt-35-turbo \
--rate_limit 50 \
--output_file data/synthetic_queries.jsonl &&

python3 src/generate_data.py --n_samples 1 \
--openai_key $OPENAI_API_KEY \
--load_from_hf \
--model gpt-35-turbo \
--rate_limit 50 \
--output_file data/synthetic_queries_test.jsonl &&


sh scripts/train.sh
