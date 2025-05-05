<p align="center"> <a href="https://www.linkedin.com/company/tuya-s-a/posts/?feedView=all" target="_blank"> <img src="https://media.licdn.com/dms/image/v2/C4E0BAQFYvI1kNS1sdw/company-logo_200_200/company-logo_200_200/0/1636494916789?e=2147483647&v=beta&t=GuGZaRp2qnnQ8-MvLhk1nbfUTAM0CUeBJAyvbgubwgg" width="120" alt="NestJS Logo" /> </a> </p> <h1 align="center">Prueba Tecnica Ingeniero Prompt</h1> <p align="center"> Prueba tecnica con los puntos 6, 7, 8, 11 y 12 de la prueba.</p>

### Tener en cuenta:
En cada una de las carpetas o modulos llamados por los puntos de la prueba, tiene un archivo __jupyter__ __.ipynb__ donde se ejecutaron los test, pueden utilizar este para realizar las pruebas paso a paso, o tambien esta el archivo de __python__ __.py__ donde esta el agente como una clase y al final de este archivo un ejemplo de funcionamiento, donde si ya tienen la apikey lista de openai en el archivo __.env__ pueden ejecutar este script. 

hay agentes que tienen piden esta key en su entrada cuando se va a ejecutar: __pdf_path__, esto hace referencia al path del PDF del extracto, se segiere colocar el path completo.

como lo mencione en la documentacion, si uno de estos agentes: DUDAS FRECUENTES & CARGOS NO RECONOCIDOS, es probado y su respuesta no es la deseada, pueda ser que sea por el formato del extracto bancario, ya que esta implementacion se trato de hacerla general pero daba algunos errores, que se pueden corregir con mas tiempo, por el momento se hizo implementacion personalizada o a la medida, que es el enfoque que busque para que el modelo no entrara en alucinaciones y sepa como es el formato del extracto bancario del cual le hare preguntas.

# 🚀 Requisitos previos
- Esta prueba tecnica esta realizada en Python, asi que se debe tener instalado un interprete que cuente con la version +3.12 
- Se esta usando los modelos de OpenAI, asi que se debe tener un APIKEY

# 🛠️ Instalación y configuración

### 1. Clonar el proyecto

```
git clone https://github.com/yaspur/PT-Ing-Prompt.git
```

### 2. Instalar dependencias

```
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

⭐ Clona el archivo __.env.template__ y renómbralo a __.env__

⭐ La variable de entorno __OPENAI_API_KEY__ para utilizar los modelos de Open AI

# 🧑‍💻 Tecnologías utilizadas

Python

Frameworks: LangChain / LangGraph