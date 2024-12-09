from openai import OpenAI
from langsmith.wrappers import wrap_openai
from langsmith import traceable, evaluate

from dotenv import load_dotenv
load_dotenv()

client = wrap_openai(OpenAI())

prompt = """
You are a supportive and encouraging AI goal-setting assistant. Your task is to help a person set and achieve their goals by guiding them through a structured process. Follow these steps carefully:

1. Begin by asking the person what goal they want to achieve. Wait for their response before proceeding.

2. Once you have the goal, ask why this goal is important to them. Encourage them to reflect deeply on their motivation and explain why it is important to understand motivation behind the goal. Give examples of possible motivations. Wait for their response before proceeding.

3. Ask when they want to achieve this goal by. Wait for their response before proceeding.

4. Ask about their starting point. What steps they have already taken toward the goal ? Wait for their response before proceeding.

5. Based on the information provided, create an initial plan. Break the main goal into smaller, manageable sub-goals. Research and suggest at least three online resources or programs that could help the person get started with their goal. Provide brief descriptions of each resource.
Ask about any potential blockers or challenges they foresee in achieving their goal. Wait for their response before proceeding.

6. Use the information about blockers to refine the initial plan. Adjust timelines if necessary and suggest strategies to overcome these challenges.

7. Create a low-friction measurement/tracking system that the person can easily maintain. Ensure that this system is directly linked to observable milestones in their goal journey.

8. Help the person integrate this new plan into their daily life. Suggest specific actions they can take each day or week to work towards their goal.

9. Explain how you will help them adjust their daily goals and assignments in response to patterns detected during their progress.

10. Throughout this process, be supportive and encouraging. Acknowledge the difficulty of change and the courage it takes to set and pursue goals.

11. Summarize the entire plan, including the goal, timeline, steps, resources, tracking system, and integration into daily life. Present this summary in a clear, organized manner.

12. Ask if they have any questions or if there's anything they'd like to adjust in the plan.

13. Conclude with words of encouragement and offer to be available for future check-ins and adjustments as they progress towards their goal.

Remember to be patient, allowing the person time to respond to each question before moving on. Use empathetic language and positive reinforcement throughout the conversation. If at any point the person seems unsure or discouraged, offer reassurance and help them break down their goals or challenges into smaller, more manageable parts.

Begin by asking about their goal:

<goal_inquiry>What specific goal would you like to achieve? Please describe it in detail.</goal_inquiry>

"""


@traceable
def coaching_agent(inputs: dict) -> dict:
    messages = [
        {"role": "system", "content": prompt},
        *inputs["messages"]
    ]

    result = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.2
    )

    return {
        "message": {
            "role": "assistant",
            "content": result.choices[0].message.content
        }
    }

# The name or UUID of the LangSmith dataset to evaluate on.
data = "Coaching Prompts"

# A string to prefix the experiment name with.
experiment_prefix = "Basic prompt tests"

def correctness_evaluator(run, example) -> list[dict]:
    """
    Evaluates the intial response to the goal set by the user
    
    Args:
        run: Contains the run information including inputs and outputs
        example: Contains the reference example if available
    
    Returns:
        Dictionary with score (0-1) and explanation
    """
    # Extract the original LeetCode problem from inputs
    coaching_prompt = run.inputs["inputs"]["messages"][-1]["content"]
    
    # Extract the model's generated tests
    coaching_response = run.outputs["message"]["content"]
    
    # Rest of the evaluation logic remains the same
    evaluation_prompt = f"""
    Given this goal from the user:
    {coaching_prompt}

    Evaluate the response given criteria below:
    {coaching_response}
    
    IScore from 0-1:
    1 = The response asks why that goal the user provided is important
    0 = The response doesn't asks why that goal the user provided is important

    RScore from 0-1: 
    1 = The response explains why it is important to know the reason behind the goal
    0 = The response doesn't explain why it is important to know the reason behind the goal

    EScore from 0-1: 
    1 = The response contains examples of possible motivations for the goal
    0 = The response doesn't contain examples of possible motivations for the goal
    
    Return only the three numbers (0-1) for IScore, RScore and EScore.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a test evaluation assistant. Respond only with a three numbers 0-1."},
            {"role": "user", "content": evaluation_prompt}
        ],
        temperature=0
    )
    
    try:
        scores = response.choices[0].message.content.strip().split()
        if len(scores) != 3:
            raise ValueError("Expected three scores")
        iscore = int(scores[0])
        rscore = int(scores[1])
        escore = int(scores[2])
        return [{
            "key": "iscore",
            "score": iscore,  
            "explanation": f"Question presence: {iscore}"
        },{
            "key": "rscore",
            "score": rscore,  
            "explanation": "Explanation presence"
        },{
            "key": "escore",
            "score": escore,  
            "explanation": "Examples presence"
        }]
    except ValueError:
        return [{
            "key": "iscore",
            "score": 0,
            "explanation": "Failed to parse score"
        },{
            "key": "rscore",
            "score": 0,
            "explanation": "Failed to parse score"
        },{
            "key": "escore",
            "score": 0,
            "explanation": "Failed to parse score"
        }]

# List of evaluators to score the outputs of target task
evaluators = [
    correctness_evaluator
]

# Evaluate the target task
results = evaluate(
    coaching_agent,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix
)