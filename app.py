import gradio as gr
from agent import financial_agent
from utils import summarize_to_dataframe, plot_confidence_trend
from agent import update_memory

# Function to handle user input and display results
def run_agent_interface(user_prompt, rag_context):
    result = financial_agent(user_prompt, rag_context)

    # Save to persistent memory
    update_memory(user_prompt, result["action_plan"])

    # Format output
    df = summarize_to_dataframe(result["summaries"])
    plot = plot_confidence_trend(conversation_history)

    return result["action_plan"], result["evaluation"], df, plot

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 📈 Financial News Agent with Gemini")

    user_prompt = gr.Textbox(label="Your Prompt", placeholder="e.g. Should I invest in Tesla this week?")
    rag_input = gr.Textbox(label="Domain Context (Optional)", placeholder="e.g. Tesla Q1 earnings beat estimates...")

    run_button = gr.Button("Analyze")

    with gr.Row():
        action_output = gr.Textbox(label="📊 Investment Plan")
        eval_output = gr.Textbox(label="🧪 Evaluation")

    df_output = gr.Dataframe(label="Summarized News")
    chart_output = gr.Plot(label="Confidence Trend")

    run_button.click(fn=run_agent_interface, 
                     inputs=[user_prompt, rag_input],
                     outputs=[action_output, eval_output, df_output, chart_output])

# Launch Gradio UI
demo.launch()
