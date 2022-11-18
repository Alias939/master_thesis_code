from transformers import AutoModel, AutoTokenizer
import torch
import scenario_existing_dataset_creator
from transformers import TrainingArguments, Trainer
from dataloader import IterableDataset, ValidationDataset
import torch.distributed as dist
from collections import OrderedDict
from dataclasses import dataclass, field
model = AutoModel.from_pretrained('clips/mfaq')
tokenizer = AutoTokenizer.from_pretrained('clips/mfaq')

def collate_fn(batch):
    questions, answers, page_ids = [], [], []
    print(batch)

    for item in batch:


        questions.append(item['question'])

        answers.append(item['answer'])
        page_ids.append(item["id"])


    output = tokenizer(
        questions + answers,
        return_tensors="pt",
        pad_to_multiple_of=8,
        padding = 'max_length',
        truncation = True,
    )
    #output["id"] = torch.Tensor(page_ids)
    print(output)
    return output

# dataset = scenario_existing_dataset_creator.create_existing_scenario_dataset()
# tokenized_dataset = dataset.map(collate_fn, batched=True)
# train_dataset = tokenized_dataset.shuffle(seed=42)
from datasets import load_dataset
dataset = load_dataset("clips/mfaq", 'nl')
print(dataset['train'])


train_dataset = scenario_existing_dataset_creator.create_existing_scenario_dataset()
train_dataset = IterableDataset(train_dataset,'nl')

from transformers import TrainingArguments




def distributed_softmax(q_output, a_output, rank, world_size):
    q_list = [torch.zeros_like(q_output) for _ in range(world_size)]
    a_list = [torch.zeros_like(a_output) for _ in range(world_size)]
    dist.all_gather(tensor_list=q_list, tensor=q_output.contiguous())
    dist.all_gather(tensor_list=a_list, tensor=a_output.contiguous())
    q_list[rank] = q_output
    a_list[rank] = a_output
    q_output = torch.cat(q_list, 0)
    a_output = torch.cat(a_list, 0)
    return q_output, a_output


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output["last_hidden_state"] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        page_id = inputs.pop("page_id", None)
        outputs = model(**inputs)
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])
        q_logits, a_logits = torch.chunk(sentence_embeddings, 2)
        if self.args.distributed_softmax and self.args.local_rank != -1 and return_outputs is False:
            q_logits, a_logits = distributed_softmax(
                q_logits, a_logits, self.args.local_rank, self.args.world_size
            )
            labels = torch.arange(q_logits.size(0), device=a_logits.device)
        cross_entropy = torch.nn.CrossEntropyLoss()
        dp = q_logits.mm(a_logits.transpose(0, 1))
        labels = torch.arange(dp.size(0), device=dp.device)
        loss = cross_entropy(dp, labels)
        #print(OrderedDict({"q_logits": q_logits, "a_logits": a_logits, "page_id": page_id}))
        if return_outputs:
            outputs = OrderedDict({"q_logits": q_logits, "a_logits": a_logits, "page_id": page_id})
        return (loss, outputs) if return_outputs else loss

@dataclass
class CustomTrainingArgument(TrainingArguments):
    distributed_softmax: bool = field(default=False)

training_args = TrainingArguments(max_steps=10,output_dir="test_trainer",evaluation_strategy="no",distributed_softmax=distributed_softmax())

trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    args = training_args

)
trainer.train()