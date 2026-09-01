import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()

class GroqLLM:
    def __init__(self, model_name: str = "openai/gpt-oss-120b", api_key: str =None):
        """
        Initialize Groq LLM
        
        Args:
            model_name: Groq model name (qwen2-72b-instruct, llama3-70b-8192, etc.)
            api_key: Groq API key (or set GROQ_API_KEY environment variable)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass api_key parameter.")
        
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        
        print(f"Initialized Groq LLM with model: {self.model_name}")

    def generate_response(self, query: str, context: str, max_length: int = 500) -> str:
        """
        Generate response using retrieved context
        
        Args:
            query: User question
            context: Retrieved document context
            max_length: Maximum response length
            
        Returns:
            Generated response string
        """
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an expert assistant specializing in ISRO missions and space technology.
                    Answer the user's question using ONLY the information provided in the context below.

                    Instructions:
                    - Write the answer as a single, well-structured paragraph.
                    - Do not use bullet points, numbering, or headings.
                    - Summarize the information naturally instead of copying the context verbatim.
                    - If the context is in a different language other than English, reply with "I couldn't understand the Language"
                    - If the context does not contain enough information to answer the question, reply with: "I couldn't find sufficient information in the provided ISRO documents."
                    Context:
                    {context}
                    Question: {question}
                    Answer: Provide a clear and informative answer based on the context above. If the context doesn't contain enough information to answer the question, say so."""
                            )
        
        # Format the prompt
        formatted_prompt = prompt_template.format(context=context, question=query)
        
        try:
            # Generate response
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
        