# This is a main script that tests the functionality of specific agents.
# It requires no user input.

import random
from aios.scheduler.fifo_scheduler import FIFOScheduler
from aios.utils.utils import parse_global_args, delete_directories
from pyopenagi.agents.agent_factory import AgentFactory
from pyopenagi.agents.agent_process import AgentProcessFactory
import warnings
from aios.llm_kernel import llms
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import find_dotenv, load_dotenv

def clean_cache(root_directory):
    targets = {'.ipynb_checkpoints', '__pycache__', ".pytest_cache", "context_restoration"}
    delete_directories(root_directory, targets)

#control is only true 5% of the time
def control():
    return random.random() < 0.05

def main():
    # parse arguments and set configuration for this run accordingly
    warnings.filterwarnings("ignore")
    parser = parse_global_args()
    args = parser.parse_args()

    llm_name = args.llm_name
    max_gpu_memory = args.max_gpu_memory
    eval_device = args.eval_device
    max_new_tokens = args.max_new_tokens
    scheduler_log_mode = args.scheduler_log_mode
    agent_log_mode = args.agent_log_mode
    llm_kernel_log_mode = args.llm_kernel_log_mode
    _use_backend = args.use_backend
    load_dotenv()


    #test gpt-4 and ollama

    llm_models_names = ['gpt-4', 'ollama/llama3']
    llm_models = {}
    for model in llm_models_names:
        llm_models[model] = llms.LLMKernel(
        llm_name=model,
        max_gpu_memory=max_gpu_memory,
        eval_device=eval_device,
        max_new_tokens=max_new_tokens,
        log_mode=args.llm_kernel_log_mode,
        use_backend=_use_backend
    )


    scheduler = FIFOScheduler( #
        llm=llm_models['gpt-4'],
        log_mode = scheduler_log_mode
    )

    agent_process_factory = AgentProcessFactory()

    agent_factory = AgentFactory(
        llm = llm_models['gpt-4'],
        agent_process_queue = scheduler.agent_process_queue,
        agent_process_factory = agent_process_factory,
        agent_log_mode = agent_log_mode
    )

    agent_thread_pool = ThreadPoolExecutor(max_workers=500)

    scheduler.start()

    topics = ["F1 Tournaments", "Machine Learning", "Quantum Computing", "Climate Change", "Blockchain Technology"]


    agent_tasks = []

    #mxm matrix 
    epoch = 1
    for _ in range(epoch):
        for model_A in llm_models_names: #model A is the generator
            for model_B in llm_models_names: #model B is the detector
                topic = random.choice(topics)
                if (control()): #control group
                    truthlie_task_input = f"Generate a set of three statements where all three statements are true about the topic: {topic}."
                else:
                    truthlie_task_input = f"Generate a set of three statements where two are true and one is a lie about the topic: {topic}."
                scheduler.set_llm(llm_models[model_A])

                truthlie_agent = agent_thread_pool.submit(
                    agent_factory.run_agent, 
                    "TruthLie", truthlie_task_input,
                )

                truthlie_result = truthlie_agent.result()  # Get the result from TruthLie agent
                scheduler.set_llm(llm_models[model_B])

                guesser_task_input = f"Here are three statements about {topic}: {truthlie_result['result']} Guess which one is the lie. If you think none of choices is a false statement, please respond with None of the Above. "
                guesser_agent = agent_thread_pool.submit(
                    agent_factory.run_agent, 
                    "Guesser", guesser_task_input
                )

                agent_tasks.extend([truthlie_agent, guesser_agent])

    for r in as_completed(agent_tasks):
        _res = r.result()
        print(_res)  # Print the result from each agent

    scheduler.stop()

    clean_cache(root_directory="./")

if __name__ == "__main__":
    main()
