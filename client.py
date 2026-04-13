from swift import InferRequest, InferClient, RequestConfig, InferStats

engine = InferClient(host='127.0.0.1', port=8000)
print(f'models: {engine.models}')
metric = InferStats()
request_config = RequestConfig(max_tokens=512, temperature=0)

# 这里使用了3个infer_request来展示batch推理
# 支持传入本地路径、base64和url
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': '<image><image>两张图的区别是什么？'}],
                 images=['/root/autodl-tmp/zz/datasets/test/latex_ocr_1.jpg',
                        '/root/autodl-tmp/zz/datasets/test/latex_ocr_1.jpg']),
    InferRequest(messages=[{'role': 'user', 'content': '<image>convert the image to latex code'}],
                 images=['/root/autodl-tmp/zz/datasets/test/latex_ocr_1.jpg']),
]

resp_list = engine.infer(infer_requests, request_config, metrics=[metric])
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
print(f'response2: {resp_list[2].choices[0].message.content}')
print(metric.compute())
metric.reset()
