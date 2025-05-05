import os

from typing import Dict, List, Annotated, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

from PyPDF2 import PdfReader

from dotenv import load_dotenv
load_dotenv()


class AgentCargosNoReconocidos:
    
    class OverallState(TypedDict):
        prompt: str
        pdf_path: str
        instrucciones: str
        texto_pdf: str
        texto_estructurado: dict
        context: dict
        resultado: str
        
    def __init__(self, model="gpt-4o-mini"):
        
        graph = StateGraph(self.OverallState)

        # Añadir nodos
        graph.add_node("pdf_to_text", self.pdf_to_text)
        graph.add_node("structure_text", self.structure_text)
        graph.add_node("process_user_question", self.process_user_question)

        graph.add_edge(START, "pdf_to_text")
        graph.add_edge("pdf_to_text", "structure_text")
        graph.add_edge("structure_text", "process_user_question")
        graph.add_edge("process_user_question", END)
                
        self.graph = graph.compile().with_config({"run_name": "agente_cargos_no_reconocidos"})
        
        self.llm = ChatOpenAI(
            model=model, 
            temperature=0, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ) 
        
    
    def pdf_to_text(self, state: OverallState) -> OverallState:
        """Extraer el texto de un PDF"""
        
        pdf_path = state['pdf_path']
        
        pdfreader = PdfReader(fr"{pdf_path}")
        
        raw_text = ""
        
        for page in pdfreader.pages:
            raw_text += page.extract_text()
        
        state['texto_pdf'] = raw_text
        
        return state
    
    def structure_text(self, state: OverallState) -> OverallState:
        """Estructurar el texto extraido del PDF"""
        
        class Transacciones(BaseModel):
            fecha_transaccion: str = Field(..., description="Fecha del extracto")
            valor_transaccion: int = Field(..., description="Valor de la extracto")
            tipo: Literal["Gasto", "Ingreso"] = Field(..., description="Tipo de transacción determinado si tiene un - o +")
            doc: int = Field(..., description="Número de documento de la transacción")
            clase_movimiento: str = Field(..., description="Clase de movimiento de la transacción")
            oficina: str = Field(..., description="Oficina a la que pertenece la transacción")
            
        
        class DatosExtracto(BaseModel):
            fecha_extracto: str = Field(..., description="Fecha del extracto")
            tipo_cuenta: str = Field(..., description="Tipo de cuenta")
            numero_cuenta: float = Field(..., description="Frecuencia de patrones repetitivos")
            nombre_cliente: float = Field(..., description="Nivel de coherencia temática y de flujo")
            saldo_anterior: float = Field(..., description="Variedad en estructura de oraciones")
            mas_creditos: float = Field(..., description="Variedad de vocabulario")
            menos_debitos: float = Field(..., description="Uso de conectores y transiciones")
            nuevo_saldo: float = Field(..., description="Uso de ejemplos y anécdotas")
            saldo_promedio: float = Field(..., description="Uso de metáforas y analogías")
            saldo_total_bolsillo: float = Field(..., description="Uso de preguntas retóricas")
            transacciones: List[Transacciones] = Field(..., description="Lista de transacciones hechas en el extracto")
            
        llm = self.llm
        
        llm = llm.bind_tools(tools=[DatosExtracto], tool_choice="DatosExtracto")
        parser = JsonOutputKeyToolsParser(key_name="DatosExtracto", first_tool_only=True)
        
        prompt_template = """
        Analiza el siguiente texto que es un extracto bancario y extrae la informacion de una forma estructurada.
        El extracto como texto es el siguiente: 
        
        <texto>
        {texto}
        </texto>
        
        La información que debes extraer es la siguiente:
        - Fecha del extracto
        - Tipo de cuenta
        - Número de cuenta
        - Nombre del cliente
        - Saldo anterior
        - Más créditos
        - Menos débitos
        - Nuevo saldo
        - Saldo promedio
        - Saldo total en bolsillo
        
        - Lista de transacciones hechas en el extracto con los siguientes campos:
            - Fecha de la transacción
            - Valor de la transacción
            - Tipo de transacción (determinado si tiene un - fue un gasto o + fue un ingreso)
            - doc 
            - Clase de movimiento
            - Oficina a la que pertenece la transacción
        """
        
        prompt = PromptTemplate.from_template(
            template=prompt_template,
            partial_variables={
                "texto": state['texto_pdf']
            },
        )
        
        chain = prompt | llm | parser
        
        response = chain.invoke({})["transacciones"]
        
        return {"context": response}
    
    def process_user_question(self, state: OverallState) -> OverallState:
        """Procesar y dar respuesta a la pregunta del usuario"""
        
        class Respuesta(BaseModel):
            respuesta: str = Field(..., description="Respuesta a la pregunta del usuario acerca del extracto")
        
        llm = self.llm
        
        llm = llm.bind_tools(tools=[Respuesta], tool_choice="Respuesta")
        parser = JsonOutputKeyToolsParser(key_name="Respuesta", first_tool_only=True)
        
        system_message = state['prompt']
        
        human_question = """\
        El cliente ha hecho la siguiente pregunta acerca de un cargo no reconocido en su extracto::
        <pregunta>
        {instrucciones}
        </pregunta>
        
        La información que tienes para responder es la siguiente:
        <contexto>
        {context}
        </contexto>
        """
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template(human_question, partial_variables={
                "context": state['context'],
                "instrucciones": state['instrucciones']
            }),
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages)
        
        chain = prompt | llm | parser
        
        response = chain.invoke({})["respuesta"]
        
        return {"resultado": response}
    
if __name__ == "__main__":
    
    agent = AgentCargosNoReconocidos()
    
    with open('prompt_no_CoT.txt', 'r', encoding='utf-8') as f:
        prompt_no_cot = f.read()
        
    with open('prompt_CoT.txt', 'r', encoding='utf-8') as f:
        prompt_cot = f.read()
        
    # Define el estado inicial
    initial_state = {
        "pdf_path": "C:\pt_ing_prompt\data\EXTRACTO_portafolio2025033120018008910224_unlocked.pdf",
        "instrucciones": "¿La transaccion con numero de doc 0913 o 913 de cuanto fue su valor?",
        "prompt": prompt_cot
    }
    
    # Ejecuta el agente
    result = agent.app.invoke(initial_state)
    print(result)