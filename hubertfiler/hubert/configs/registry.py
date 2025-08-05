DATA_DICT = {
    "pretrain": "/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/camilo-fineweb-top600M",
    "finetune": "/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/books",
    "eval": "/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/finewebedu-val",
    "score": "/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/camilo-finewebedu-10B"
}

MODEL_DICT = {
    "hubert": "/n/netscratch/sham_lab/Everyone/cbrownpinilla/models/HUBERt-resumed/resume-HUBERT/hubert_14132897/final-model.pt",
    "hubert_conditional": "/n/netscratch/sham_lab/Everyone/cbrownpinilla/models/finetune-hubert/finetune-HUBERT/hubert_14233456/model/model-step5000.pt",
    "roberta": "/n/netscratch/sham_lab/Everyone/cbrownpinilla/models/roberta-new/roBERTa-new-dataloader-h100/hubert_13761623/final-model.pt",
    "roberta_conditional": "/n/netscratch/sham_lab/Everyone/cbrownpinilla/models/finetune-roberta-new/finetune-roBERTa/hubert_14234555/model/model-step5000.pt",
    "hubert-resume": "/n/netscratch/sham_lab/Everyone/cbrownpinilla/models/bert-new/HUBERT-new-loader/hubert_13704341/model/model-step300000.pt" 
}