"""Shared frozen inputs, resident RGB images and comparable CPU-visible timing."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src')]
from vlm_runtime import load_processor
from vlmbench.metrics import quality


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Inputs:
    def __init__(self,split):
        self.folder=ROOT/'data/processed/docvqa' if split=='development' else HERE/'data'
        file=self.folder/('development.jsonl' if split=='development' else 'evaluation.jsonl')
        self.file=file;self.rows=[json.loads(s) for s in file.read_text().splitlines()]
        self.cfg=read(ROOT/'configs/protocol.lock.json')['runtime']
        self.cfg['dataset_path']=str(self.folder.relative_to(ROOT))
        self.processor=load_processor(self.cfg)
        self.images={}
        for row in self.rows:
            key=row['pixel_sha256']
            if key not in self.images:
                with Image.open(self.folder/row['image_path']) as image:self.images[key]=image.convert('RGB')
                assert hashlib.sha256(self.images[key].tobytes()).hexdigest()==key
        self.eos=[151645,151643]

    def prompt(self,row):
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['question']+'\n'+self.cfg['smoke']['prompt_suffix']}]}]
        return self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)

    def prepare_cpu(self,processor,rows,cfg):
        images=[self.images[r['pixel_sha256']] for r in rows]
        texts=[self.prompt(r) for r in rows]
        cpu=processor(text=texts,images=images,padding=True,return_tensors='pt')
        counts=(cpu['image_grid_thw'].prod(-1)//processor.image_processor.merge_size**2).tolist()
        lengths=cpu['attention_mask'].sum(-1).tolist()
        assert all(n<=768 for n in counts) and all(n+64<=1024 for n in lengths)
        return cpu,{'input_lengths':lengths,'visual_tokens':counts,'grid_thw':cpu['image_grid_thw'].tolist(),
                    'pixel_shape':list(cpu['pixel_values'].shape)}

    def expected(self,row):
        import torch
        cpu,meta=self.prepare_cpu(self.processor,[row],self.cfg)
        return {'question_id':row['question_id'],'input_tokens':meta['input_lengths'][0],
                'grid_thw':meta['grid_thw'],'prompt_sha256':hashlib.sha256(cpu['input_ids'].numpy().tobytes()).hexdigest(),
                'pixel_bf16_sha256':hashlib.sha256(cpu['pixel_values'].to(torch.bfloat16).view(torch.uint8).numpy().tobytes()).hexdigest()}

    def scenarios(self,seed):
        grouped=defaultdict(list)
        for row in self.rows:grouped[row['page_id']].append(row)
        pages=[q for q in grouped.values() if len(q)>=2][:16]
        hot=[r for page in pages for r in page[1:]]
        random.Random(seed).shuffle(hot)
        order=list(grouped);random.Random(seed).shuffle(order)
        session=[r for page in order for r in grouped[page]]
        unique=[grouped[page][0] for page in order]
        return [('hot',[page[0] for page in pages],hot),('session',[],session),('unique',[],unique)]
