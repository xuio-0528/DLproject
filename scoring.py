import jsonlines
import json

predict = []
with jsonlines.open('./preds/mbpp/cleaned/predict_check_zeroshot_512_mbpp_QAformat.jsonl') as reader:
    for obj in reader:
        predict.append(obj)

with open('./preds/mbpp/zeroshot_QAformat_score.jsonl', encoding= "utf-8",mode="w") as writer:      
    total_error = 0
    for row in predict:
        code = row['pred']
        code = code.strip()
        function_name = row['test_list'][7:row['test_list'].find("(")]
        code = code.replace(code[4:code.find("(")], function_name)
        code = code + '\n' + row['test_list']
        error=0
        try:
            exec(code)
        except:
            error=1
            total_error+=1
        if error==0:
            line = {'completion' : code.replace('\n', '\\n'), 'right' : 1}
        else:
            line = {'completion' : code.replace('\n', '\\n'), 'right' : 0}
        writer.write(json.dumps(line) + "\n")
    print(total_error)