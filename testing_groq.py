from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.llms import Ollama

llm = Ollama(model="llama3.1")

def llamaconfig(article_text):
    comparison_prompt = PromptTemplate(
        input_variables=["article_text"],
        template="""
        Go through each line in Target Text and extract all possible entities without missing any paragraph, and categorize each entity type precisely. 
        Include persons, locations, organizations, dates, times, quantities, and any other relevant entities. exclude time and quantity

        Target Text: {article_text}
        """
    )

    chain = comparison_prompt | llm

    # Run the comparison chain
    result = chain.invoke({
        "article_text": article_text
    })

    # Output the result
    print("Extraction Result:", result)
    
    # Return the content from the result
    # The result from a chain is typically a string or a dict with 'content' key
    if isinstance(result, str):
        return result
    elif isinstance(result, dict) and 'content' in result:
        return result['content']
    elif hasattr(result, 'content'):
        return result.content
    else:
        return str(result)  # Fallback to string representation