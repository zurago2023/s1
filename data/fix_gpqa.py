import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd()))))
import glob as glob
from datasets import load_dataset
import random
random.seed(42)
import ast
from tqdm import tqdm
from datasets import Dataset

TEMPLATE = "{Question}\nAnswer Choices:\n(A) {choice1}\n(B) {choice2}\n(C) {choice3}\n(D) {choice4}"

def process_example(example):
    if example['source_type'] == 'Idavidrein/gpqa':
        metadata = ast.literal_eval(example['metadata'])
        choices = [
            metadata["Incorrect Answer 1"].strip('\n'),
            metadata["Incorrect Answer 2"].strip('\n'),
            metadata["Incorrect Answer 3"].strip('\n'),
            metadata["Correct Answer"].strip('\n'),
        ]
        random.shuffle(choices)
        correct_answer_index = choices.index(metadata["Correct Answer"].strip('\n'))
        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": f"{chr(65 + correct_answer_index)}",
        }
        question = TEMPLATE.format(Question=example["question"], choice1=out_doc["choice1"], choice2=out_doc["choice2"], choice3=out_doc["choice3"], choice4=out_doc["choice4"])
        solution = example["solution"] + "\n\n" + "Answer: " + out_doc["answer"]
        example['question'] = question
        example['solution'] = solution  
        return example
    else:
        return example

if __name__ == "__main__":
    dataset = load_dataset("s1/s1K")['train']
    new_dataset = []
    for doc in tqdm(dataset):
        new_dataset.append(process_example(doc))
    new_dataset = Dataset.from_list(new_dataset)
    new_dataset.push_to_hub("s1/s1K")