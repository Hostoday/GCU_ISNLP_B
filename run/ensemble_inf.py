
import argparse
import json
import tqdm
from sentence_transformers import CrossEncoder
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from src.ens_data import CustomDataset,DataCollatorForInferenceDataset


# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--device", type=str, required=True, help="device to load the model")
g.add_argument("--concat_file_list", type = str, nargs='+', required=True , help = "concat file others")

# fmt: on


def main(args):
    model = CrossEncoder('ddobokki/electra-small-sts-cross-encoder')
    
    dataset = CustomDataset(args.concat_file_list)
    #print(dataset)
    data_collator = DataCollatorForInferenceDataset()
    train_dataloader = DataLoader(dataset,batch_size=1,collate_fn=data_collator)
    
    with open("resource/data/일상대화요약_test.json", "r") as f:
        result = json.load(f)
    
    batch_start_idx = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
            outputs = batch["input_ids"]
            sts_score = []
            for output in outputs[0]:
                score = 0
                for candidate in outputs[0]:
                    score += model.predict([output,candidate])
                sts_score.append(score)
            sts_score = torch.tensor([sts_score])
            val, idx = torch.max(sts_score,1)
            result[batch_start_idx]["output"] = outputs[0][idx.item()]
            print(result[batch_start_idx]["output"])
            batch_start_idx += 1
            

    with open(f"./inference{args.output}", "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))