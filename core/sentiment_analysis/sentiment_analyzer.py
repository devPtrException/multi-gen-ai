import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer
import gradio as gr

model_path = "/home/mrmauler/DRIVE/projects/gen-ai/multi/models/models--distilbert--distilbert-base-uncased-finetuned-sst-2-english/snapshots/714eb0fa89d2f80546fda750413ed43d93601a13"
sentiment_analyzer_pipeline = pipeline("sentiment-analysis", model=model_path)


def sentiment_bar_chart(df):
    sentiment_counts = df["Sentiment"].value_counts()

    # Create a bar chart
    fig, ax = plt.subplots()
    sentiment_counts.plot(kind="pie", ax=ax, autopct="%1.1f%%", color=["green", "red"])
    ax.set_title("Review Sentiment Counts")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    # ax.set_xticklabels(['Positive', 'Negative'], rotation=0)

    # Return the figure object
    return fig


def read_reviews_and_analyze_sentiment(file_object):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_object)

    # Check if 'Review' column is in the DataFrame
    if "Reviews" not in df.columns:
        raise ValueError("Excel file must contain a 'Review' column.")

    # Apply the get_sentiment function to each review in the DataFrame
    df["Sentiment"] = df["Reviews"].apply(sentiment_analyzer)
    chart_object = sentiment_bar_chart(df)
    return df, chart_object


def analyze_sentiment(input):
    output = sentiment_analyzer_pipeline(input)
    formatted_output = f"Sentiment: {output[0]['label']}\nConfidence: {str(output[0]['score'])[2:4]}%"  # 0.9998725652694702
    return formatted_output


def sentiment_analyzer(input):
    sentiment = sentiment_analyzer_pipeline(input)
    return sentiment[0]["label"]


def main():

    # text = "that was a great movie"
    # result = analyze_sentiment(text)
    # print(result)

    gr.close_all()

    #     interface = gr.Interface(
    #         fn=analyze_sentiment,
    #         inputs=gr.Textbox(label="Enter sentiment to analyze ...", lines=5),
    #         outputs=gr.Textbox(label="Summary ...", lines=2),
    #         title="Summarizer",
    #         description="Summarizes a gives text corpus",
    #     )
    #     interface.launch()
    #
    #     demo = gr.Interface(
    #         fn=read_reviews_and_analyze_sentiment,
    #         inputs=[gr.File(file_types=["xlsx"], label="Upload your review comment file")],
    #         outputs=[gr.Dataframe(label="Sentiments"), gr.Plot(label="Sentiment Analysis")],
    #         title="@GenAILearniverse Project 3: Sentiment Analyzer",
    #         description="THIS APPLICATION WILL BE USED TO ANALYZE THE SENTIMENT BASED ON FILE UPLAODED.",
    #     )
    #     demo.launch()

    # Interface 1: Text sentiment analysis
    text_interface = gr.Interface(
        fn=analyze_sentiment,
        inputs=gr.Textbox(label="Enter sentiment to analyze", lines=5),
        outputs=gr.Textbox(label="Summary", lines=2),
        title="Text Sentiment Analyzer",
        description="Analyzes sentiments based on a given text corpus.",
    )

    # Interface 2: File-based sentiment analysis
    file_interface = gr.Interface(
        fn=read_reviews_and_analyze_sentiment,
        inputs=[gr.File(file_types=[".xlsx"], label="Upload your review/comment file")],
        outputs=[gr.Dataframe(label="Sentiments"), gr.Plot(label="Sentiment Analysis")],
        title="File-Based Sentiment Analyzer",
        description="Analyzes sentiments based on uploaded Excel files.",
    )

    # Combine both interfaces in tabs
    full_interface = gr.TabbedInterface(
        interface_list=[text_interface, file_interface],
        tab_names=["Analyze Text", "Analyze File"],
    )

    full_interface.launch()


if __name__ == "__main__":
    main()


# result = read_reviews_and_analyze_sentiment("../Files/Prod-review.xlsx")
# print(result)
# Example usage:
# df = read_reviews_and_analyze_sentiment('path_to_your_excel_file.xlsx')
# print(df)


# Example usage:
# Assuming you have a dataframe `df` with appropriate data
# fig = sentiment_bar_chart(df)
# fig.show()  # This line is just to visualize the plot in a local environment
