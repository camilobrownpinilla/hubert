from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling



# Make data loaders
# For right now might be hardcoded for finefineweb, but might fix later?
def get_loaders(cfg):
    train, test, val = _get_loaders('train', cfg), _get_loaders('test', cfg), _get_loaders('validation', cfg)
    collator = DataCollatorForLanguageModeling(tokenizer=cfg.tokenizer)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, collate_fn=collator, num_workers=1)
    test_loader = DataLoader(test, batch_size=cfg.batch_size, collate_fn=collator, num_workers=1)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, collate_fn=collator, num_workers=1)
    return train_loader, test_loader, val_loader

def _get_loaders(split: str, cfg):
    assert split in ['train', 'test', 'validation'], "Choose one of 'train', 'test', 'validation'"
    key = f'-{split}' if split in ['test', 'validation'] else ''
    ds = load_dataset('m-a-p/FineFineWeb'+key, streaming=True, split='train').remove_columns(['language_score', 'date', 'url', 'file_path', 'dump', 'global_id', 'lang', 'domain', 'token_count'])
    ds = ds.with_format(type='torch')
    ds = ds.map(lambda ex: cfg.tokenizer(ex['text'], 
                                                   truncation=True, 
                                                   padding=True,
                                                   add_special_tokens=True), 
                            batched=True, remove_columns=['text'])
    return ds

def get_toy_loaders(cfg):
    train, test, val = _get_toy_loaders('train', cfg), _get_toy_loaders('test', cfg), _get_toy_loaders('validation', cfg)
    collator = DataCollatorForLanguageModeling(tokenizer=cfg.tokenizer)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, collate_fn=collator, num_workers=1, shuffle=False)
    test_loader = DataLoader(test, batch_size=cfg.batch_size, collate_fn=collator, num_workers=1, shuffle=False)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, collate_fn=collator, num_workers=1, shuffle=False)
    return train_loader, test_loader, val_loader

def _get_toy_loaders(split: str, cfg):
    assert split in ['train', 'test', 'validation'], "Choose one of 'train', 'test', 'validation'"
    key = f'-{split}' if split in ['test', 'validation'] else ''
    # Take one batch of data
    ds = load_dataset('m-a-p/FineFineWeb'+key, streaming=True, split='train').remove_columns(['language_score', 'date', 'url', 'file_path', 'dump', 'global_id', 'lang', 'domain', 'token_count']).take(cfg.batch_size)
    ds = ds.with_format(type='torch')
    ds = ds.map(lambda ex: cfg.tokenizer(ex['text'], 
                                                   truncation=True, 
                                                   padding=True,
                                                   add_special_tokens=True), 
                            batched=True, remove_columns=['text'])
    return ds