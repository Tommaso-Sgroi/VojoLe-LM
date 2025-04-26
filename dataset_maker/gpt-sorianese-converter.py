import json
import os
from time import sleep
from dataset_maker.utils import clean_gpt_output_formatted_text
# import tqdm
from openai import OpenAI

from dataset_maker.job_server import Database

RATE_LIMIT_RPM = 50 # RPM
RATE_LIMIT_RPD = 10_000 # RPM


OPEN_AI_KEY = os.getenv('OPENAI_API_KEY')
if OPEN_AI_KEY is None or OPEN_AI_KEY.strip() == '':
    print(OPEN_AI_KEY)
    raise Exception("No OPENAI_API_KEY provided")

database = Database()
client = OpenAI(api_key=OPEN_AI_KEY)

with open("./data/raw_resources/generation_prompt2.1.txt", 'r') as f:
    SYSTEM_PROMPT = f.read()


def load_samples():
    from datasets import load_dataset
    fw = load_dataset("../data/fineweb-2/data/ita_Latn", split="test", columns=["text", "id"])

    # for entry in fw:
    #     phrases.append({"id": entry["id"], "text": entry["text"]})


def create_batch_file():
    global client, database, RATE_LIMIT_RPM
    items = []
    for _ in range(RATE_LIMIT_RPM):
        custom_key, content = database.get_next_item()
        item = {
            "custom_id": str(custom_key),
            "method": "POST", "url": "/v1/chat/completions",
            "body": {
                # "model": "gpt-4.1-mini",
                "model": "gpt-4.1-mini",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content}
                ],
                "max_tokens": 5000,
                "logprobs": True,

            }
        }
        items.append(item)

    with(open('/tmp/batchinput.jsonl', 'w', encoding='utf-8')) as f:
        for i in range(len(items)):
            f.write(json.dumps(items[i]) + '\n')

    batch_input_file = client.files.create(
        file=open("/tmp/batchinput.jsonl", "rb"),
        purpose="batch"
    )
    return batch_input_file, items


def retrieve_batch_output():
    # sleep(5*60) # sleep 5 minutes
    global database, client
    batches = database.get_pending_batch_jobs()

    if len(batches) == 0: return -1

    print('trying to retrieve results')
    for batch_id in reversed(batches):
        batch_job = client.batches.retrieve(batch_id)
        if batch_job.completed_at is not None:
            result_file_id = batch_job.output_file_id
            result = client.files.content(result_file_id).content

            result_file_name = f"data/output/gpt-4.1-translation.jsonl"
            with open(result_file_name, 'ab') as file:
                file.write(result)

            # # Loading data from saved file
            with open(result_file_name, 'r') as file:
                for line in file:
                    # Parsing the JSON string into a dict and appending to the list of results
                    res = json.loads(line.strip())
                    text = res['response']['body']['choices'][0]['message']['content']
                    custom_id = res['custom_id']

                    text = clean_gpt_output_formatted_text(text)
                    database.insert_translation(custom_id, text)
                print('-' * 500)
            database.update_batch_job(batch_id, 1)

        elif batch_job.failed_at is not None:
            print('Error: job failed ' + str(batch_job.output_file_id))
            database.update_batch_job(batch_id, -1)
    return 1


def batch_inference():
    global database
    batches = []
    # for _ in tqdm.tqdm(range(RATE_LIMIT_RPD)):
    if True:
        try:
            batch_input_file, items = create_batch_file()
            print('batch file created')
            batch_input_file_id = batch_input_file.id

            batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "Italian - Sorianese dialect translation"
                }
            )
            print('batch job created')

            batches.append(batch)
            it = [int(i['custom_id']) for i in items]
            database.add_batch_job(batch.id, it)
        except Exception as e:
            raise e




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='gpt-translator',
        )
    parser.add_argument('--retrieve_batch', nargs='?', type=int, default=None, help='Retrieve the output from the batch API, if the argument is n > 0 then it try to retrieve the batch size every n seconds.')
    parser.add_argument('--continuous_retrieve_batch', action='store_true', default=False, help='Do not exit if there are no pending jobs.')
    parser.add_argument('--batch_inference', nargs='+', type=int, help='Schedule a batch job with N size')

    args = parser.parse_args()

    if args.batch_inference:
        batch_inference()
        pass
    if args.retrieve_batch is not None:

        while database.has_pending_jobs() > 0 or args.continuous_retrieve_batch:
            code = retrieve_batch_output()
            if code == -1: print('No batch retrieved')
            else: print('Batch retrieved')
            print("Completion: ", str(database.completion_status() * 100) + "%")
            print('Trying in: ', args.retrieve_batch, 'sec', '\n', '_'*200, '\n')
            sleep(args.retrieve_batch)
    print('Done')
