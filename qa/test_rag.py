from qa.rag_qa import rag_qa

if __name__ == "__main__":
    question = "What research gaps exist in transformer interpretability?"
    result = rag_qa(question)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nCITATIONS:\n")
    print(result["citations"])
