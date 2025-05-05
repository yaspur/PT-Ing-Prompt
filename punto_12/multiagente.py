import os

from typing import TypedDict, Literal


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

from punto_7.agente_dudas_frecuentes import AgentDudasFrecuentes
from punto_8.agente_cargos_no_reconocidos import AgentCargosNoReconocidos
from punto_11.agente_detector_IA import DetectorIA

from dotenv import load_dotenv
load_dotenv()

class MultiAgentSystem:
    
    class OverallState(TypedDict):
        instrucciones: str
        agente: str
        resultado: str
        pdf_path: str
        prompt: str
        
    def __init__(self, model="gpt-4o-mini"):
        
        graph = StateGraph(self.OverallState)
        
        # Definir los agentes
        agente_cargos_no_reconocidos = AgentCargosNoReconocidos()
        agente_dudas_frecuentes = AgentDudasFrecuentes()
        agente_detector_ia = DetectorIA()

        # Añadir nodos
        graph.add_node("agent_decition", self.agent_decition)
        graph.add_node("agente_cargos_no_reconocidos", agente_cargos_no_reconocidos.graph)
        graph.add_node("agente_preguntas_frecuentes", agente_dudas_frecuentes.graph)
        graph.add_node("agente_detector_ia", agente_detector_ia.graph)

        graph.add_edge(START, "agent_decition")

        graph.add_conditional_edges("agent_decition", self.route_agent, {
            "cargos_no_reconocidos": "agente_cargos_no_reconocidos",
            "preguntas_frecuentes": "agente_preguntas_frecuentes",
            "detector_ia": "agente_detector_ia"
        })

        graph.add_edge("agente_cargos_no_reconocidos", END)
        graph.add_edge("agente_preguntas_frecuentes", END)
        graph.add_edge("agente_detector_ia", END)
                
        self.graph = graph.compile().with_config({"run_name": "multi_agent_system"})
        
        self.llm = ChatOpenAI(
            model=model, 
            temperature=0, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ) 
        
    def agent_decition(self, state: OverallState) -> OverallState:
        """Segun el query del usuario rutearlo al agente mas apropiado"""
        
        class Agent(BaseModel):
            agent: Literal["cargos_no_reconocidos", "preguntas_frecuentes", "detector_ia"] = Field(..., description="Agente al que se le asigna la tarea") 
        
        llm = self.llm
        
        llm = llm.bind_tools(tools=[Agent], tool_choice="Agent")
        parser = JsonOutputKeyToolsParser(key_name="Agent", first_tool_only=True)
        
        prompt_template = """
        Analiza el siguiente texto o query ingresado por un usuario y determina a qué agente de los siguientes se le debe asignar la tarea:
        
        - Cargos no reconocidos: Agente especializado en resolver problemas relacionados con cargos o transaccciones no reconocidos en extractos bancarios.
        - Preguntas frecuentes: Agente especializado en responder preguntas frecuentes sobre el extracto bancario.
        - Detector de IA: Agente especializado en detectar si el texto fue generado por una IA o por un humano.
        
        El texto o query es el siguiente:
        <texto>
        {texto}
        </texto>
        """
        
        prompt = PromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"texto": state['instrucciones']}
        )
        
        chain = prompt | llm | parser
        
        response = chain.invoke({})["agent"]
        
        return {"agente": response}
    
    def route_agent(self, state: OverallState) -> str:
        """Determinar el agente al que se le asigna la tarea"""
        
        agent = state["agente"]
        
        if agent == "cargos_no_reconocidos":
            # Asignar la tarea al agente de cargos no reconocidos
            return "cargos_no_reconocidos"
        
        elif agent == "preguntas_frecuentes":
            # Asignar la tarea al agente de preguntas frecuentes
            return "preguntas_frecuentes"
        
        elif agent == "detector_ia":
            # Asignar la tarea al agente detector de IA
            return "detector_ia"
        
    
if __name__ == "__main__":
    
    agent = MultiAgentSystem()
    
    with open('punto_8\prompt_no_CoT.txt', 'r', encoding='utf-8') as f:
        prompt_no_cot = f.read()
        
    with open('punto_8\prompt_CoT.txt', 'r', encoding='utf-8') as f:
        prompt_cot = f.read()
        
    # Define el estado inicial
    initial_state = {
        "instrucciones": "¿La transaccion con numero de doc 0913 o 913 de cuanto fue su valor?",
        "pdf_path": "C:\pt_ing_prompt\data\EXTRACTO_portafolio2025033120018008910224_unlocked.pdf",
        "prompt": prompt_cot
    }
    
    # Ejecuta el agente
    result = agent.app.invoke(initial_state)
    print(result)
        