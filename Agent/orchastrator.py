

from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding


from langchain_core.tools import tool
from langchain_core.messages import SystemMessage,HumanMessage,ToolMessage, AIMessage
from langgraph.graph import StateGraph,START,END,MessagesState
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_groq import ChatGroq
from tensorflow.keras.models import load_model
from langchain_chroma import Chroma
from dotenv import load_dotenv
import json, joblib, requests
import numpy as np, pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq
from typing import Optional, Literal,TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
import requests
from langchain_core.messages import AIMessageChunk
from datetime import date
import os



# ================= STEP 2: config and shared resources =================
load_dotenv()

# ---- paths (built from this file's location, so it runs from any folder)
BASE_DIR   = Path(__file__).resolve().parent.parent      # Agent/ -> project root
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data" / "raw"
VECTOR_DIR = str(BASE_DIR / "vectorstore" / "chroma")

# ---- constants
CITIES = ["Delhi", "Pune", "Kolkata"]
CITY_COORDS = {
    "Delhi":   (28.576448, 77.18678),
    "Pune":    (18.523726, 73.86876),
    "Kolkata": (22.601053, 88.31776),
}
DATA_START = "2015-01-01"
DATA_END   = "2025-01-01"

EMBED_MODEL       = "BAAI/bge-small-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
COLLECTION        = "weather_kb"

# ---- embedding model (same settings as the notebook)
class FastEmbedBGE(Embeddings):
    """bge-small through ONNX: same vectors as the PyTorch version, no torch needed."""
    def __init__(self, model_name, query_instruction, cache_dir):
        self.model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        self.query_instruction = query_instruction

    def embed_documents(self, texts):
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text):
        return next(iter(self.model.embed([self.query_instruction + text]))).tolist()


embedding_model = FastEmbedBGE(
    EMBED_MODEL,
    QUERY_INSTRUCTION,
    cache_dir=str(MODELS_DIR / "fastembed_cache"),
)


# ---- open the existing Chroma store (do NOT rebuild it here)
vectorstore = Chroma(
    collection_name=COLLECTION,
    persist_directory=VECTOR_DIR,
    embedding_function=embedding_model,
)

# ---- LSTM model, scaler and its settings
lstm_model = load_model(str(MODELS_DIR / "lstm_weather_model.keras"))
scaler = joblib.load(MODELS_DIR / "scaler.pkl")
with open(MODELS_DIR / "config.json") as f:
    config = json.load(f)

FEATURES = config["features"]      # column order matters
WINDOW   = config["window"]        # 30
HORIZON  = config["horizon"]       # 7

# to convert a scaled temp_max prediction back to degrees C
T_MIN   = scaler.data_min_[0]
T_RANGE = scaler.data_max_[0] - scaler.data_min_[0]


# ================= STEP 3: search_kb tool with filters =================
TOP_K = 3

@tool
def search_kb(
    query: str,
    city: Optional[Literal["Delhi", "Pune", "Kolkata"]] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    doc_type: Optional[Literal[
        "monthly_summary", "seasonal_summary", "event_log", "event_summary",
        "city_profile", "records", "trends", "rainy_season", "reference", "dataset_card",
    ]] = None,
) -> str:
    """
    Search the weather knowledge base for Delhi, Pune and Kolkata (data from 2015 to 1 Jan 2025).
    Use it for past weather: monthly and seasonal summaries, heatwaves, cold waves, heavy rain,
    records, yearly trends, city climate profiles, and definitions of weather terms.
    It cannot give forecasts.

    Arguments:
    - query: what to look for, in plain words.
    - city: Delhi, Pune or Kolkata. Always set it when the question names a city.
    - year: a 4-digit number such as 2024. Only monthly_summary, seasonal_summary and event_log
      documents have a year; do not set it for city_profile, records, trends or rainy_season.
    - month: 1 to 12. Only monthly_summary documents have a month; do not set it for other types.
    - doc_type: narrow the search to one kind of document (optional).

    Returns a JSON list of passages, each with 'source', 'similarity' and 'text'.
    """
    # build the Chroma filter from whichever arguments were given
    conditions = []
    if city is not None:
        conditions.append({"city": city})
    if year is not None:
        conditions.append({"year": year})
    if month is not None:
        conditions.append({"month": month})
    if doc_type is not None:
        conditions.append({"doc_type": doc_type})

    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    results = vectorstore.similarity_search_with_score(query, k=TOP_K, filter=where)
    if not results:
        return "No documents found for these filters."

    passages = []
    for doc, distance in results:
        passages.append({
            "source": doc.metadata["path"],
            "similarity": round(1 - distance, 3),
            "text": doc.page_content,
        })
    return json.dumps(passages, ensure_ascii=False)




@tool
def get_forecast_lstm(city: Literal["Delhi", "Pune", "Kolkata"]) -> str:    
    """
    this tool predic the accurate 7-day weather forecast for Delhi, Pune or Kolkata.
    Returns a JSON table with 'date' and 'pred_temp_max_C'.
    "temperature forecast only (daily maximum)". It gives no rain, wind or minimum temperature and other informations

    Arguments:
    - city: The city for which to get the forecast (Delhi, Pune, or Kolkata).

    """
   
   
    lat, lon = CITY_COORDS[city]
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={ 
           "latitude": lat, "longitude": lon, "timezone": "GMT", 
           "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max," 
                    "relative_humidity_2m_mean,pressure_msl_mean,weathercode", 
           "past_days": WINDOW - 1, "forecast_days": 1,        # 29 past days + today = 30 rows 
       }, timeout=30) 

    
    df = pd.DataFrame(r.json()["daily"]) 

    df.columns = ["date"] + FEATURES 
    assert len(df) == WINDOW 
    
    
    X = scaler.transform(df[FEATURES]).reshape(1, WINDOW, len(FEATURES)) 
    pred_scaled = lstm_model.predict(X, verbose=0)[0]            # 7 values, scaled 0-1 
    t_min, t_range = scaler.data_min_[0], scaler.data_max_[0] - scaler.data_min_[0] 
    pred_c = pred_scaled * t_range + t_min                  # back to °C 
    
    last_date = pd.Timestamp(df["date"].iloc[-1])

    days = []

    for i in range(1, HORIZON + 1):
        next_date = last_date + pd.Timedelta(days=i)
        days.append(next_date.date())

    return json.dumps(
        {
            "date":[str(date) for date in days],
            "pred_temp_max_c":[round(temp,1) for temp in pred_c.tolist()]
        }
    )




    
def find_place(name):
    """(lat, lon, label) for a place name, or None. Our 3 cities use the stored coordinates."""
    if name in CITY_COORDS:
        lat, lon = CITY_COORDS[name]
        return lat, lon, name
    r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                     params={"name": name, "count": 1, "language": "en", "format": "json"}, timeout=15)
    results = r.json().get("results")
    if not results:
        return None
    top = results[0]
    label = ", ".join(x for x in [top["name"], top.get("admin1"), top.get("country")] if x)
    return top["latitude"], top["longitude"], label
    
    
    

@tool
def get_forecast_api(city:str, days:int)->str:
    """
    this tool capable to predict the weather forecast like rain,wind humidity and presssure not suitable for temperature prediction of Pune, Delhi and Kolkata city only other than that it can predict for all cities.
    
    Arguments:
    -city: The city for which to get the forecast the weather example other any city than Delhi, Pune or Kolkata because LSTM Model will predict it according.
    -days: The number of days for which to get the forecast (1-7).
    Returns a JSON table with 'date', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum_mm', 'windspeed_10m_max', 'relative_humidity_2m_mean', 'pressure_msl_mean', 'weathercode'
    The first row is TODAY. days=1 gives today only; for tomorrow use days=2 and read the second row.
    """

    if days<1 or days>7:
        return "days must be between 1 and 7"
    
    
    
    
    
    place = find_place(city)
    if place is None:
        return f"I could not find a place called '{city}'."
    lat, lon, label = place
    
    url=f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&timezone=GMT&forecast_days={days}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,relative_humidity_2m_mean,pressure_msl_mean,weathercode"
    response=requests.get(url,timeout=30)

    df=pd.DataFrame(response.json()["daily"])
    df.columns=["date","temperature_2m_max","temperature_2m_min","precipitation_sum_mm","windspeed_10m_max","relative_humidity_2m_mean","pressure_msl_mean","weathercode"]
    df["date"]=pd.to_datetime(df["date"])


    return json.dumps(
        {
            "date": [str(date) for date in df["date"].tolist()],
            "temperature_2m_max": [round(temp,1) for temp in df["temperature_2m_max"].tolist()],
            "temperature_2m_min": [round(temp,1) for temp in df["temperature_2m_min"].tolist()],
            "precipitation_sum_mm": [round(precip,1) for precip in df["precipitation_sum_mm"].tolist()],
            "windspeed_10m_max": [round(wind,1) for wind in df["windspeed_10m_max"].tolist()],
            "relative_humidity_2m_mean": [round(humidity,1) for humidity in df["relative_humidity_2m_mean"].tolist()],
            "pressure_msl_mean": [round(pressure,1) for pressure in df["pressure_msl_mean"].tolist()],
        }
    )


    
@tool
def ask_user(question: str) -> str:
    """
    Ask the user a short question when a detail needed for another tool is missing
    (for example which city: Delhi, Pune,Kolkata or other city). Use this instead of guessing.
    Returns the user's answer as text.
    """
    return interrupt(question)      


tools=[search_kb,get_forecast_lstm,get_forecast_api,ask_user]


llm=ChatGroq(
     model="openai/gpt-oss-120b",
    temperature=0,
    reasoning_format="parsed",
    max_retries=2,
    streaming=True,
)

llm_with_tools=llm.bind_tools(tools)




tool_node = ToolNode(tools, handle_tool_errors=True)

class MessagesState(TypedDict):
     messages: Annotated[list, add_messages]
    
    


    
SYSTEM_TEXT = (
    "You are a weather assistant for cities. Use the tools for every number and keep answers short. "
    "If the user did not provide enough information for tool calling (or another detail a tool needs), call ask_user to ask. Never guess a any value that needed for tool calling."
    "also You are AI BOT Assistance who can help user to solve his general queries but make sure you are suppose to solve user query not to perform any task which are unrelavenet to your capabilities. if user ask simple query related then answer if it ask any other different questions like coding or any othre domain like games or Medical or other domain simply answer that you are not capable to solve this query and you are only capable to solve weather related queries, bt make sure that your tasks is also  to use engage with user if he asks some questions on it. your tasks is not only for the weather also some engagement for user. to understand betterly some technical term related to weather as ChatGPT does"
    "Past weather (search_kb) and the get_forecast_lstm temperature forecast exist only for Delhi, Pune and Kolkata. get_forecast_api works for any city in the world."
    "and if output is unstructured make in in structured format user readable, same for table related data too"
)    

def chatbot(state:MessagesState)->dict:
    messages=state["messages"]
    response = llm_with_tools.invoke([SystemMessage(content=f"Today's date is {date.today()}. " + SYSTEM_TEXT)] + messages)

    return {"messages":[response]}



graphBuilder=StateGraph(
    MessagesState,
)

graphBuilder.add_node("chatbot",chatbot)
graphBuilder.add_node("tools",tool_node)

graphBuilder.add_edge(START,"chatbot")

graphBuilder.add_conditional_edges(
    "chatbot",
    tools_condition
)

graphBuilder.add_edge("tools", "chatbot")

checkpointer=MemorySaver()

graph=graphBuilder.compile(checkpointer=checkpointer)
config={"configurable":{"thread_id":"thread_1"}}




def run_agent(question, thread_id="thread_1"):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    response = ""
    if graph.get_state(config).interrupts:                 # waiting for an answer: this text is the answer
        inputs = Command(resume=question)
    else:
        inputs = {"messages": [HumanMessage(content=question)]}

    for mode, data in graph.stream(inputs, config, stream_mode=["messages", "updates"]):
        if mode == "messages":                             # collect the model's text
            chunk, meta = data
            if meta["langgraph_node"] == "chatbot" and chunk.content:
                response += chunk.content
        else:                                              # tool events (shown on the server console)
            for node, update in data.items():
                if node == "chatbot":
                    for call in update["messages"][-1].tool_calls:
                        if call["name"] != "ask_user":
                            print(f"[calling {call['name']} {call['args']}]")
                elif node == "tools":
                    for m in update["messages"]:
                        print(f"[{m.name} finished]")

    paused = graph.get_state(config).interrupts            # ask_user stopped the graph
    if paused:
        return {"status": "needs_input", "question": paused[0].value}
    return {"status": "done", "answer": response}


def orchastrator(message: str, thread_id: str = "thread_1") -> dict:
    return run_agent(message, thread_id)







