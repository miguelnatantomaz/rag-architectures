from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    return ChatPromptTemplate.from_template("""
    Responda a pergunta com base apenas no contexto abaixo:

    <context>
    {context}
    </context>

    Pergunta: {question}
    """)