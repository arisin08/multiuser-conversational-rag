import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from multi_user_rag.evaluation.evaluator import EvaluationService
from inference import InferenceService

def main():
    inference_service=InferenceService()
    evalutor= EvaluationService()

    user_id="Ari_08"
    question="What is Machine Learning?"

    ground_truth=(
                    "Machine learning is a field of AI that enables systems to learn "
                    "patterns from data and improve performance without explicit programming."
                 )
    
    #Running Inference
    print("\nRunning Inference\n")
    result=inference_service.invoke_response(user_id=user_id, prompt=question)

    answer=result["answer"] 
    context=result["context"] 

    #Running Evaluation
    print("\nRunning Evaluation\n")
    evalution_metrics=evalutor.evaluate_all(
                                             question=question, 
                                             rag_output=answer, 
                                             ground_truth=ground_truth, 
                                             retrieved_context=context     
                                           )         
    
    print("\nEvaluation Results:\n")
    for metric_name, metric_result in evalution_metrics.items():
        print(f"{metric_name}:")
        print(f"  Score: {metric_result['Score']}")
        print(f"  Reason: {metric_result['Reason']}")
        print()


if __name__ == "__main__":
    main()