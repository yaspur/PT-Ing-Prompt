import os

from typing import Dict, List, Annotated, TypedDict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

class DetectorIA:
    
    class OverallState(TypedDict):
        instrucciones: str
        analisis_preliminar: str
        caracteristicas: dict
        patrones_detectados: List[str]
        conclusion: str
        score_ia: int  # Score de 0 a 100
        resultado: str
        
    def __init__(self, model="gpt-4o-mini"):
        
        graph = StateGraph(self.OverallState)

        graph.add_node("analisis_preliminar_texto", self.analisis_preliminar_texto)
        graph.add_node("extraer_caracteristicas", self.extraer_caracteristicas)
        graph.add_node("detectar_patrones", self.detectar_patrones)
        graph.add_node("tomar_decision", self.tomar_decision)


        graph.add_edge(START, "analisis_preliminar_texto")
        graph.add_edge(START, "extraer_caracteristicas")
        graph.add_edge(START, "detectar_patrones")

        graph.add_edge("analisis_preliminar_texto", "tomar_decision")
        graph.add_edge("extraer_caracteristicas", "tomar_decision") 
        graph.add_edge("detectar_patrones", "tomar_decision")

        graph.add_edge("tomar_decision", END)
        
        self.graph = graph.compile().with_config({"run_name": "agente_detector_IA"})
        
        self.llm = ChatOpenAI(
            model=model, 
            temperature=0, 
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ) 
        
    def analisis_preliminar_texto(self, state: OverallState) -> OverallState:
        """Realiza un análisis preliminar del texto a evaluar."""
        
        llm = self.llm
        
        prompt_template = """
        Analiza el siguiente texto y describe tus impresiones iniciales sobre si podría haber sido generado por IA:
        
        TEXTO A ANALIZAR:
        "{texto}"
        
        Considera aspectos como:
        - Naturalidad del lenguaje
        - Estructura y ritmo
        - Presencia de patrones repetitivos
        - Uso de frases genéricas o plantillas comunes
        
        Proporciona solo un análisis breve y objetivo.
        """
        
        prompt = PromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"texto": state['texto']}
        )
        
        chain = prompt | llm
        
        response = chain.invoke({})
            
        return {"analisis_preliminar": response.content}
    

    def extraer_caracteristicas(self, state: OverallState) -> OverallState:
        """Extrae características lingüísticas cuantificables del texto."""
        
        class Carateristicas(BaseModel):
            repeticiones: float = Field(..., description="Frecuencia de patrones repetitivos")
            coherencia: float = Field(..., description="Nivel de coherencia temática y de flujo")
            complejidad_sintactica: float = Field(..., description="Variedad en estructura de oraciones")
            diversidad_lexica: float = Field(..., description="Variedad de vocabulario")
        
        llm = self.llm
        
        llm = llm.bind_tools(tools=[Carateristicas], tool_choice="Carateristicas")
        parser = JsonOutputKeyToolsParser(key_name="Carateristicas", first_tool_only=True)
        
        prompt_template = """
        Analiza el siguiente texto y calcula los siguientes indicadores en una escala de 0.0 a 1.0:
        
        TEXTO A ANALIZAR:
        "{texto}"
        
        1. Repeticiones: Frecuencia de patrones repetitivos (0=sin repeticiones, 1=altamente repetitivo)
        2. Coherencia: Nivel de coherencia temática y de flujo (0=incoherente, 1=perfectamente coherente)
        3. Complejidad sintáctica: Variedad en estructura de oraciones (0=muy simple, 1=muy compleja)
        4. Diversidad léxica: Variedad de vocabulario (0=muy limitada, 1=muy diversa)
        """
        
        prompt = PromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"texto": state['texto']}
        )
        
        chain = prompt | llm | parser
        
        response = chain.invoke({})
        
        caracteristicas_dict = {
            "repeticiones": response["repeticiones"],
            "coherencia": response["coherencia"],
            "complejidad_sintactica": response["complejidad_sintactica"],
            "diversidad_lexica": response["diversidad_lexica"]
        }
        
        return {"caracteristicas": caracteristicas_dict}
        
    
    def detectar_patrones(self, state: OverallState) -> OverallState:
        """Detecta patrones específicos comunes en textos generados por IA."""
        
        class Patrones(BaseModel):
            patrones: List[str] = Field(..., description="Lista de patrones detectados")
        
        llm = self.llm
        
        llm = llm.bind_tools(tools=[Patrones], tool_choice="Patrones")
        parser = JsonOutputKeyToolsParser(key_name="Patrones", first_tool_only=True)
        
        prompt_template = """
        Analiza el siguiente texto e identifica la presencia de patrones típicos de texto generado por IA.
        
        TEXTO A ANALIZAR: 
        "{texto}"
        
        Busca específicamente:
        1. Frases introductorias genéricas ("Es importante señalar que", "Cabe destacar que")
        2. Estructuras enumerativas excesivas o listas
        3. Neutralidad excesiva o falta de posicionamiento
        4. Uso de frases de transición predecibles
        5. Redundancias o repetición de ideas
        6. Exceso de calificativos o adverbios
        7. Conclusiones genéricas o vacías
        8. Explicaciones demasiado didácticas
        
        Devuelve SOLO una lista de los patrones detectados (sin explicaciones adicionales).
        """
        
        prompt = PromptTemplate.from_template(
            template=prompt_template,
            partial_variables={"texto": state['texto']}
        )
        
        chain = prompt | llm | parser
        
        response = chain.invoke({})["patrones"]
        
        return {"patrones_detectados": response}
    
    
    def tomar_decision(self, state: OverallState) -> OverallState:
        """Integra todos los análisis y asigna un score de 0-100 (0=humano, 100=IA)."""
        
        class Score(BaseModel):
            score: int = Field(..., description="Score de 0 a 100 para determinar la probabilidad de IA (0=humano, 100=IA)")
            conclusion: str = Field(..., description="Conclusión detallada sobre el análisis")
        
        llm = self.llm
        
        llm = llm.bind_tools(tools=[Score], tool_choice="Score")
        parser = JsonOutputKeyToolsParser(key_name="Score", first_tool_only=True)
        
        # Preparar el resumen de las características
        caracteristicas = state["caracteristicas"]
        patrones = "\n".join([f"- {p}" for p in state["patrones_detectados"]])
        
        system_prompt = """\
        Eres un experto en análisis de textos. Tu tarea es evaluar la probabilidad de que un texto haya sido generado por IA.
        Utiliza la información proporcionada para llegar a una conclusión fundamentada, toma el tiempo necesario para analizar\
        los resultados obtenidos previamente y llegar a una conclusion.
        
        Previamente has realizado un análisis preliminar del texto, has extraído características lingüísticas cuantificables\
        y has detectado patrones que son típicos de textos generados por IA.
        
        Tendras en cuenta los datos anteriores para llegar a:
        - Conclusiones y puntuación final
        
        ## Proporciona un score de 0 a 100, donde:
        - 0-20: Casi con certeza escrito por humano
        - 21-40: Probablemente escrito por humano
        - 41-60: Indeterminado
        - 61-80: Probablemente generado por IA
        - 81-100: Casi con certeza generado por IA
        
        El score debe ser un número entero exacto entre 0 y 100.
        
        ## Estructura tu respuesta claramente con estas dos secciones:
        No incluyas ninguna otra información o comentario adicional.
        - Conclusión: Explicación detallada de tu razonamiento
        - Score: Un número entero entre 0 y 100 que representa la probabilidad de que el texto haya sido generado por IA.
        """
        
        human_instructions = f"""\
        Según el análisis previo realizado a un texto, evalúa la probabilidad de que el texto haya sido generado por IA.
        
        Estas son las características del texto:
        
        ANÁLISIS PRELIMINAR:
        {state["analisis_preliminar"]}
        
        CARACTERÍSTICAS LINGÜÍSTICAS:
        - Repeticiones: {caracteristicas["repeticiones"]}
        - Coherencia: {caracteristicas["coherencia"]}
        - Complejidad sintáctica: {caracteristicas["complejidad_sintactica"]}
        - Diversidad léxica: {caracteristicas["diversidad_lexica"]}
        
        PATRONES DETECTADOS:
        {patrones}
        """
        
        texto = state["texto"]
        
        human_prompt = f"""\
        Este es el texto que se analizo y se obtuvieron las características anteriores:
        
        <texto>
        "{texto}"
        </texto>
        """
        
        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(human_instructions),
            HumanMessagePromptTemplate.from_template(human_prompt)
        ]
        
        prompt = ChatPromptTemplate.from_messages(messages=messages)
        
        chain = prompt | llm | parser
        
        response = chain.invoke({})
        
        state["conclusion"] = response["conclusion"]
        state["score_ia"] = response["score"]
        
        state["resultado"] = f"El texto tiene una probabilidad de {response['score']}% de haber sido generado por IA. Conclusión: {response['conclusion']}"
        
        return state


if __name__ == "__main__":
    
    detector = DetectorIA()
    
    # Ejemplo de uso
    texto = "Este es un ejemplo de texto que podría haber sido generado por IA."
    
    result = detector.app.invoke({"instrucciones": texto})
    
    print(result)