"""
RNA序列的Tokenizer
将RNA序列（A, U, G, C）转换为模型输入
"""

import torch
from typing import List, Union


class RNATokenizer:
    """RNA序列tokenizer，将碱基转换为token ID"""
    
    def __init__(self):
        # RNA的4种碱基 + 特殊token
        self.vocab = {
            '<PAD>': 0,  # 填充
            '<UNK>': 1,  # 未知
            '<CLS>': 2,  # 序列开始
            '<SEP>': 3,  # 序列结束
            'A': 4,      # 腺嘌呤
            'U': 5,      # 尿嘧啶
            'G': 6,      # 鸟嘌呤
            'C': 7,      # 胞嘧啶
        }
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        
    def encode(
        self, 
        sequence: Union[str, List[str]], 
        max_length: int = 512,
        add_special_tokens: bool = True,
        return_tensors: bool = False
    ):
        """
        将RNA序列编码为token ID
        
        Args:
            sequence: RNA序列字符串或列表
            max_length: 最大序列长度
            add_special_tokens: 是否添加特殊token
            return_tensors: 是否返回torch tensor
            
        Returns:
            编码后的token ID
        """
        if isinstance(sequence, str):
            sequences = [sequence]
        else:
            sequences = sequence
            
        encoded_sequences = []
        
        for seq in sequences:
            # 转换为大写
            seq = seq.upper().replace('T', 'U')  # DNA转RNA
            
            # 编码序列
            tokens = []
            if add_special_tokens:
                tokens.append(self.vocab['<CLS>'])
                
            for base in seq:
                if base in self.vocab:
                    tokens.append(self.vocab[base])
                else:
                    tokens.append(self.vocab['<UNK>'])
                    
            if add_special_tokens:
                tokens.append(self.vocab['<SEP>'])
                
            # 截断或填充
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.vocab['<PAD>']] * (max_length - len(tokens))
                
            encoded_sequences.append(tokens)
            
        if return_tensors:
            return torch.tensor(encoded_sequences, dtype=torch.long)
        
        return encoded_sequences
    
    def decode(self, token_ids: Union[List[int], torch.Tensor]) -> str:
        """
        将token ID解码为RNA序列
        
        Args:
            token_ids: token ID列表或tensor
            
        Returns:
            RNA序列字符串
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()
            
        sequence = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, '<UNK>')
            if token not in ['<PAD>', '<CLS>', '<SEP>', '<UNK>']:
                sequence.append(token)
                
        return ''.join(sequence)
    
    def batch_encode(
        self,
        sequences: List[str],
        max_length: int = 512,
        add_special_tokens: bool = True
    ):
        """
        批量编码RNA序列
        
        Args:
            sequences: RNA序列列表
            max_length: 最大序列长度
            add_special_tokens: 是否添加特殊token
            
        Returns:
            编码后的batch
        """
        encoded = self.encode(
            sequences,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors=True
        )
        
        # 创建attention mask
        attention_mask = (encoded != self.vocab['<PAD>']).long()
        
        return {
            'input_ids': encoded,
            'attention_mask': attention_mask
        }
