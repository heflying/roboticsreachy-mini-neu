
# OpenCompass 评测配置文件
# 自动生成，请勿手动修改

datasets = [

    dict(
        type='MultiChoiceDataset',
        path=r'E:\programs\Robot\Reachy\LLM评价标准研究\eval_opencompass\datasets\03_智力_中文特有知识_val.jsonl',
        name='03_智力_中文特有知识',
        abbr='03_智力_中文特有知识',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer'
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template=dict(
                    begin=[
                        dict(role='SYSTEM', fallback_role='HUMAN', 
                             prompt='You are an AI assistant for elderly care. Be concise, gentle, and patient.')
                    ],
                    round=[
                        dict(role='HUMAN', prompt='Please answer the following multiple-choice question. Think step by step, then give the final answer (A, B, C, or D).\n\n{query}\n\nFormat:\nReasoning: [your reasoning]\nAnswer:'),
                        dict(role='BOT', prompt='{answer}')
                    ]
                )
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    )
,
    dict(
        type='MultiChoiceDataset',
        path=r'E:\programs\Robot\Reachy\LLM评价标准研究\eval_opencompass\datasets\03_智力_常识推理_val.jsonl',
        name='03_智力_常识推理',
        abbr='03_智力_常识推理',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer'
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template=dict(
                    begin=[
                        dict(role='SYSTEM', fallback_role='HUMAN', 
                             prompt='You are an AI assistant for elderly care. Be concise, gentle, and patient.')
                    ],
                    round=[
                        dict(role='HUMAN', prompt='Please answer the following multiple-choice question. Think step by step, then give the final answer (A, B, C, or D).\n\n{query}\n\nFormat:\nReasoning: [your reasoning]\nAnswer:'),
                        dict(role='BOT', prompt='{answer}')
                    ]
                )
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    )
,
    dict(
        type='MultiChoiceDataset',
        path=r'E:\programs\Robot\Reachy\LLM评价标准研究\eval_opencompass\datasets\03_智力_指令遵循_val.jsonl',
        name='03_智力_指令遵循',
        abbr='03_智力_指令遵循',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer'
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template=dict(
                    begin=[
                        dict(role='SYSTEM', fallback_role='HUMAN', 
                             prompt='You are an AI assistant for elderly care. Be concise, gentle, and patient.')
                    ],
                    round=[
                        dict(role='HUMAN', prompt='Please answer the following multiple-choice question. Think step by step, then give the final answer (A, B, C, or D).\n\n{query}\n\nFormat:\nReasoning: [your reasoning]\nAnswer:'),
                        dict(role='BOT', prompt='{answer}')
                    ]
                )
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    )
,
    dict(
        type='MultiChoiceDataset',
        path=r'E:\programs\Robot\Reachy\LLM评价标准研究\eval_opencompass\datasets\03_智力_逻辑推理_val.jsonl',
        name='03_智力_逻辑推理',
        abbr='03_智力_逻辑推理',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer'
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template=dict(
                    begin=[
                        dict(role='SYSTEM', fallback_role='HUMAN', 
                             prompt='You are an AI assistant for elderly care. Be concise, gentle, and patient.')
                    ],
                    round=[
                        dict(role='HUMAN', prompt='Please answer the following multiple-choice question. Think step by step, then give the final answer (A, B, C, or D).\n\n{query}\n\nFormat:\nReasoning: [your reasoning]\nAnswer:'),
                        dict(role='BOT', prompt='{answer}')
                    ]
                )
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    )
,
    dict(
        type='MultiChoiceDataset',
        path=r'E:\programs\Robot\Reachy\LLM评价标准研究\eval_opencompass\datasets\04_回复幻觉率_事实正确性_val.jsonl',
        name='04_回复幻觉率_事实正确性',
        abbr='04_回复幻觉率_事实正确性',
        reader_cfg=dict(
            input_columns=['question', 'A', 'B', 'C', 'D'],
            output_column='answer'
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type='PromptTemplate',
                template=dict(
                    begin=[
                        dict(role='SYSTEM', fallback_role='HUMAN', 
                             prompt='You are an AI assistant for elderly care. Be concise, gentle, and patient.')
                    ],
                    round=[
                        dict(role='HUMAN', prompt='Please answer the following multiple-choice question. Think step by step, then give the final answer (A, B, C, or D).\n\n{query}\n\nFormat:\nReasoning: [your reasoning]\nAnswer:'),
                        dict(role='BOT', prompt='{answer}')
                    ]
                )
            ),
            retriever=dict(type='ZeroRetriever'),
            inferencer=dict(type='GenInferencer'),
        ),
        eval_cfg=dict(
            evaluator=dict(type='AccEvaluator'),
        ),
    )

]

models = [

    dict(
        type='OpenAIAPI',
        path='qwen2.5:1.5b-instruct',
        name='qwen2.5_1.5b-instruct',
        api_url='http://localhost:11434/v1',
        api_key='ollama',
        max_out_len=1024,
        batch_size=1,
    )

]
