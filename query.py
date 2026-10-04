from rag import answer_query


result = answer_query(
    text="What fertilizer should I use for tomato during flowering?"
)

print("Text Query:")
print(result["query"])
print(result["answer"])
print("-----------------------------------")

result = answer_query(
    audio_filename="voicequery.mp3"
)

print("Voice Query:")
print(result["query"])
print(result["answer"])
print("-----------------------------------")