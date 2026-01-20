import os
import datetime
import sys
import asyncio
import time
import nodes
from dotenv import load_dotenv
from graph import create_graph, MAX_ITERATIONS
from rag import cyber_rag

# Load environment variables
load_dotenv()

async def main():  # Changed to async
    print("--- Start Agent System with RAG ---")
    start_time = time.time()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    result = {"messages": []}

    log_dir = "./Log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print("-Create Log Directory.")
        
    print("Successfully Run Ready")

    try:
        # Initialize RAG system
        print("Initializing RAG system...")
        await cyber_rag.initialize()
        await cyber_rag.load_cyber_data()

        cti_list = cyber_rag.cti_list
        # forecast_data = await cyber_rag.get_forecast_data()
        # Calling forecast_data in RAG. Working on LLM.
        with open("../B-MTGNN/model/Bayesian/forecast/data/",cti_list, 'r', encoding='utf-8') as f:
            cti_data = f.read().strip()
        print("RAG system ready.")

        # Execute existing Agent workflow
        app = create_graph()

        print("--- Starting Workflow (Turn: 0) ---")
        print(f"Max Iterations: {MAX_ITERATIONS}")    # Setting in graph.py
        initial_state = {
            "forecast_data": cti_data,
            "messages": [],
            "iteration_count": 0
        }

        result = await app.ainvoke(initial_state, config={"recursion_limit": 100})

        print("="*50)
        print("Execution Finished Successfully.")
        
        # Save logs
        log_path = os.path.join(log_dir, f"log-{current_time}.txt")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Log Save Time: {current_time}\n{"="*80}\n")
            for message in result.get("messages", []):
                f.write(f"{message}\n{"="*80}\n")
        print(f"Save to: {log_path}")
        print("="*50)

        end_time = time.time()
        run_time = end_time - start_time
        minutes = int(run_time // 60)
        seconds = run_time % 60
        print(f"Total run time: {minutes}min {seconds:.2f}sec")

    except Exception as e:
        print(f"\nError during execution: {e}")
        print("Ensure Ollama is running and the Language Model is available.")

        end_time = time.time()
        run_time = end_time - start_time
        minutes = int(run_time // 60)
        seconds = run_time % 60
        print(f"Total run time: {minutes}min {seconds:.2f}sec")

    finally:
        # Clean up RAG
        if cyber_rag.rag:
            await cyber_rag.finalize()

if __name__ == "__main__":
    asyncio.run(main())  # Execute as async
