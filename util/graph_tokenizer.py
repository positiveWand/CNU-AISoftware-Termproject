import torch

class GraphTokenizer:
    def __init__(self):
        self.shapes = ['sphere', 'cylinder', 'cube', 'torus', 'cone']
        self.colors = ['red', 'blue', 'green', 'yellow', 'black']
        self.x_rels = ['L', 'R', 'Sx']
        self.y_rels = ['F', 'B', 'Sy']
        self.z_rels = ['T', 'U', 'Sz']
        self.itoc = { id:color for id, color in enumerate(self.colors) }
        self.ctoi = { color:id for id, color in enumerate(self.colors) }
        self.itoxr = { id:x_rel for id, x_rel in enumerate(self.x_rels) }
        self.xrtoi = { x_rel:id for id, x_rel in enumerate(self.x_rels) }
        self.itoyr = { id:y_rel for id, y_rel in enumerate(self.y_rels) }
        self.yrtoi = { y_rel:id for id, y_rel in enumerate(self.y_rels) }
        self.itozr = { id:z_rel for id, z_rel in enumerate(self.z_rels) }
        self.zrtoi = { z_rel:id for id, z_rel in enumerate(self.z_rels) }

        self.special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    
    def encode_token(self, token):
        token_id = -1
        try:
            if token in self.special_tokens:
                token_id = self.special_tokens.index(token)
            elif len(token) == 2:
                shape = token[0]
                if shape >= len(self.shapes): raise 'Invalid shape id: ' + shape
                color = token[1]
                
                token_id = len(self.special_tokens)
                token_id += len(self.colors) * shape + self.ctoi[color]
            elif len(token) == 3:
                x_rel = token[0]
                y_rel = token[1]
                z_rel = token[2]

                token_id = len(self.special_tokens)
                token_id += len(self.shapes) * len(self.colors)
                token_id += len(self.y_rels) * len(self.z_rels) * self.xrtoi[x_rel] + len(self.z_rels) * self.yrtoi[y_rel] + self.zrtoi[z_rel]
            else:
                token_id = self.special_tokens.index('[UNK]')
        except:
            return self.special_tokens.index('[UNK]')
        
        return token_id
    
    def decode_token(self, token_id):
        token_id = int(token_id)
        token = None

        if token_id < len(self.special_tokens):
            # Special Token
            token = self.special_tokens[token_id]
        elif token_id < len(self.special_tokens) + len(self.shapes) * len(self.colors):
            # Object Token
            token_id -= len(self.special_tokens)
            shape_id = token_id // len(self.colors)
            color_id = token_id % len(self.colors)

            token = (shape_id, self.itoc[color_id])
        elif token_id < len(self.special_tokens) + len(self.shapes) * len(self.colors) + len(self.x_rels) * len(self.y_rels) * len(self.z_rels):
            # Relation Token
            token_id -= len(self.special_tokens) + len(self.shapes) * len(self.colors)
            x_rel_id = token_id // (len(self.y_rels) * len(self.z_rels))
            y_rel_id = (token_id % (len(self.y_rels) * len(self.z_rels))) // len(self.z_rels)
            z_rel_id = (token_id % (len(self.y_rels) * len(self.z_rels))) % len(self.z_rels)

            token = (self.itoxr[x_rel_id], self.itoyr[y_rel_id], self.itozr[z_rel_id])
        
        return token
    
    def encode(self, graph, add_special_tokens=False, max_length=11, padding=False):
        result = {
            'input_ids': [],
            'attention_mask': []
        }

        # (인코딩) Object Token, Relation Token -> ID
        for node in graph:
            assert len(node) == 3, 'Wrong node: ' + str(node)
            result['input_ids'].append(self.encode_token(node[0]))
            result['input_ids'].append(self.encode_token(node[1]))
            result['input_ids'].append(self.encode_token(node[2]))

        assert len(result['input_ids']) + 2 <= max_length, 'Max Length is too small!' 
        # Special Token([CLS], [SEP]) 추가하기
        if add_special_tokens:
            result['input_ids'] = result['input_ids'][:min(len(result['input_ids']), max_length - 2)]
            result['input_ids'].insert(0, self.special_tokens.index('[CLS]'))
            result['input_ids'].append(self.special_tokens.index('[SEP]'))
        else:
            result['input_ids'] = result['input_ids'][:min(len(result['input_ids']), max_length)]
        
        # Padding 추가하기
        if padding:
            for _ in range(max(0, max_length - len(result['input_ids']))):
                result['input_ids'].append(self.special_tokens.index('[PAD]'))

        # Attention Mask 생성
        for a in result['input_ids']:
            if a is not self.special_tokens.index('[PAD]'):
                result['attention_mask'].append(1)
            else:
                result['attention_mask'].append(0)
        
        return result
    
    def __call__(self, graphs, add_special_tokens=False, max_length=11, padding=False):
        result = {
            'input_ids': [],
            'attention_mask': []
        }

        for graph in graphs:
            encoded = self.encode(
                graph=graph,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding
            )
            result['input_ids'].append(encoded['input_ids'])
            result['attention_mask'].append(encoded['attention_mask'])
        
        result = {
            'input_ids': torch.tensor(result['input_ids']).long(),
            'attention_mask': torch.tensor(result['attention_mask']).long()
        }
        return result
    
    def decode(self, token_ids):
        result = []
        for token_id in token_ids:
            result.append(self.decode_token(token_id))

        return result

def token_compare(a, b):
    if type(a) == str and type(b) == str:
        if a != b:
            return False
        else:
            return True
    assert type(a) == tuple and type(b) == tuple
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True

def graph_compare(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if len(a[i]) != len(b[i]):
            return False
        for j in range(len(a[i])):
            if not token_compare(a[i][j], b[i][j]):
                return False
    
    return True

def validate_graph(graph):
    for node in graph:
        if len(node) != 3:
            return False
        if len(node[0]) != 2 or len(node[1]) != 2 or len(node[2]) != 3:
            return False
        
        if node[0][0] not in [0,1,2,3,4]:
            return False
        if node[0][1] not in ['red', 'blue', 'green', 'yellow', 'black']:
            return False
        if node[1][0] not in [0,1,2,3,4]:
            return False
        if node[1][1] not in ['red', 'blue', 'green', 'yellow', 'black']:
            return False
        if node[2][0] not in ['L', 'R', 'Sx']:
            return False
        if node[2][1] not in ['F', 'B', 'Sy']:
            return False
        if node[2][2] not in ['T', 'U', 'Sz']:
            return False
    return True

if __name__ == '__main__':
    tokenizer = GraphTokenizer()
    # 토큰 생성 후 인코딩 그리고 결과 확인
    tokens = []
    test_encode = []
    for st in tokenizer.special_tokens:
        token = st
        token_id = tokenizer.encode_token(token)
        print(token, ':', token_id)
        tokens.append(token)
        test_encode.append(token_id)
    for shape in range(5):
        for color in tokenizer.colors:
            token = (shape, color)
            token_id = tokenizer.encode_token(token)
            print(token, ':', token_id)
            tokens.append(token)
            test_encode.append(token_id)
    for x_rel in tokenizer.x_rels:
        for y_rel in tokenizer.y_rels:
            for z_rel in tokenizer.z_rels:
                token = (x_rel, y_rel, z_rel)
                token_id = tokenizer.encode_token(token)
                print(token, ':', token_id)
                tokens.append(token)
                test_encode.append(token_id)
    # 생성한거 디코딩 그리고 결과 확인
    for i in range(len(test_encode)):
        token = tokenizer.decode_token(i)
        print(i, ':', token, tokens[i], token_compare(token, tokens[i]))