import hashlib
import json
from time import time
from uuid import uuid4
from flask import Flask,jsonify,request
from textwrap import dedent


class BlockChain(object):
    def __init__(self):
        self.chain = []
        self.current_transactions = []

        #初始化创世块
        self.new_block(proof=100,previous_hash='1')

    def new_block(self, proof, previous_hash=None):
        # create a new block and add it to chain
        block={
            'index':len(self.chain)+1,
            'timestamp':time(),
            'transaction':self.current_transactions,
            'proof':proof,
            'previous_hash':previous_hash or self.hash(self.chain[-1]),
        }
        self.current_transaction=[]
        self.chain.append(block)
        return block

    def new_transaction(self,sender,recipient,amount):
        #生成一个新的交易（generate new transaction）
        #sender :发送币的地址 the address of the sender
        #recipient:接收币的地址 the address of the recipient
        #amount:币的数量
        self.current_transactions.append(
            {
                "sender":sender,
                "recipient":recipient,
                "amount":amount,
            }
        )
        return self.last_block['index']+1

    def valid_chain(self,chain):
        # 验证链是否是正确的
        last_block = chain[0]
        current_index = 1
        while current_index < len(chain):
            block = chain[current_index]
            print('{}'.format(last_block))
            print('{}'.format(block))
            print("\n-------------------\n")
            if block['previous_hash']!=self.hash(last_block):
                return False
            if not self.valid_proof(last_block['proof'],block['proof']):
                return False
            last_block = block
            current_index += 1
        return True

    def resolve_conflicts(self):
        #使用共识算法解决冲突，使用网络中最长的链
        neighbours = self.nodes
        new_chain = None
        max_length = len(self.chain)

        for node in neighbours:
            response = request.get('http://'+'{}'.format(node)+'/chain')
            if response.status_code ==200:
                length = response.json(['length'])
                chain = response.json(['chain'])

                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        if new_chain:
            self.chain = new_chain
            return True
        return False

    def proof_of_work(self,last_proof):
        proof = 0
        while self.valid_proof(last_proof,proof) is False:
            proof += 1
        return proof

    @staticmethod
    def valid_proof(last_proof,proof):
        # 验证hash(last_proof,proof)是否以'0000'开头
        # last_proof 上一个区块的proof
        # proof 当前区块的proof
        guess = '{}{}'.format(last_proof,proof).encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4]=="0000"

    @staticmethod
    def hash(block):
        # hash a block
        # 生成块的 SHA-256 hash值
        block_string = json.dumps(block,sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        # return the last block in the chain
        return self.chain[-1]


#实例化节点
app = Flask(__name__)

#为节点产生一个唯一的随机标志
node_identifier = str(uuid4()).replace('-','')

#实例化区块链
blockchain = BlockChain()

@app.route('/mine',methods=['GET'])
def mine():
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    #发送者为0表示新挖出来的新币
    #给工作量证明的节点提供奖励
    blockchain.new_transaction(
        sender='0',
        recipient=node_identifier,
        amount=1.
    )
    block = blockchain.new_block(proof)

    response  ={
        'message':"new block forged",
        'index':block['index'],
        'transactions':block['transaction'],
        'proof':block['proof'],
        'previous_hash':block['previous_hash'],
    }
    return jsonify(response),200

@app.route('/transactions/new',methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender','recipient','amount']
    if not all(k in values for k in required):
        return "miss values",400

    #create a new transaction
    index = blockchain.new_transaction(values['sender'],values['recipient'],values['amount'])
    response = {'message':'tansaction will be added to block {}'.format(index)}
    return jsonify(response),201

@app.route('/chain',methods=['GET'])
def full_chain():
    response = {
        'chain':blockchain.chain,
        'length':len(blockchain.chain),
    }
    return jsonify(response),200

@app.route('/node.register',methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error:Please supply a valid list of nodes",400
    for node in nodes:
        blockchain.register_node(node)
    response = {
        'message':'new node have been added',
        'total_nodes':list(blockchain.nodes)
    }
    return jsonify(response),201

@app.route('/node/resolve',methods=['GET'])
def consensus():
    replaced = blockchain.resolve_conflicts()

    if replaced:
        response = {
            'message':'our chain was replaced',
            'new_chain':blockchain.chain
        }
    else:
        response = {
            'message':'our chain is authoritative',
            'chain':blockchain.chain
        }
    return jsonify(response),200

if __name__ =='__main__':
    app.run(host='127.0.0.1',port=5005)