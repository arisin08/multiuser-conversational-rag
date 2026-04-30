from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics import  FaithfulnessMetric
from deepeval import evaluate
from multi_user_rag.config import (EVALUATOR_MODEL)



class EvaluationService:
    def __init__(self):
        self.threshold=0.5
        self.evaluation_model=EVALUATOR_MODEL
        self.include_reason=True
        self.verbose_mode=True

    def _format_docs(self, retrieved_context):
        if retrieved_context is None:
            return None

        # If it's already a list of strings
        if isinstance(retrieved_context, list) and all(isinstance(doc, str) for doc in retrieved_context):
            return retrieved_context

        # If it's a list of Document objects (LangChain Documents)
        try:
            return [doc.page_content for doc in retrieved_context]
        except AttributeError:
            raise ValueError(
                "retrieved_context must be a list of strings or Document objects."
            )


    def _test_case(self, question, rag_output, ground_truth=None, retrieved_context=None):
        data= {
                "input": question,
                "actual_output": rag_output
              }
        
        if ground_truth is not None:
            data["expected_output"]=ground_truth
        
        if retrieved_context is not None:
            data["retrieval_context"]=self._format_docs(retrieved_context)

        return LLMTestCase(**data)
    

    def _run_metric(self, metric, test_case):
        result=evaluate([test_case],[metric])

        return  {
                    'Score': result.test_results[0].metrics_data[0].score,
                    'Reason': result.test_results[0].metrics_data[0].reason
                }


    def evaluate_contextual_precision_metric(self, question, rag_output, ground_truth=None, retrieved_context=None):
        metric=ContextualPrecisionMetric(
                                            threshold=self.threshold,
                                            model=self.evaluation_model,
                                            include_reason=self.include_reason,
                                            verbose_mode=self.verbose_mode
                                        )
                           
        test_case=self._test_case(question, rag_output, ground_truth, retrieved_context)

        return self._run_metric(metric, test_case)


    def evaluate_contextual_recall_metric(self,question,rag_output, ground_truth, retrieved_context):
        metric=ContextualRecallMetric(
                                        threshold=self.threshold,
                                        model=self.evaluation_model,
                                        include_reason=self.include_reason,
                                        verbose_mode=self.verbose_mode
                                     )   
                    
        test_case=self._test_case(question, rag_output, ground_truth, retrieved_context)

        return self._run_metric(metric, test_case)


    def evaluate_contextual_relevancy_metric(self,question,rag_output, ground_truth, retrieved_context):
        metric=ContextualRelevancyMetric(
                                        threshold=self.threshold,
                                        model=self.evaluation_model,
                                        include_reason=self.include_reason,
                                        verbose_mode=self.verbose_mode
                                     )
        
        test_case=self._test_case(question, rag_output, ground_truth, retrieved_context)

        return self._run_metric(metric, test_case)


    def evaluate_answer_relevancy_metric(self, question, rag_output):
        metric=AnswerRelevancyMetric(
                                        threshold=self.threshold,
                                        model=self.evaluation_model,
                                        include_reason=self.include_reason,
                                        verbose_mode=self.verbose_mode
                                     )
                            
        test_case=self._test_case(question, rag_output)

        return self._run_metric(metric, test_case)
        

    def evaluate_faithfulness_metric(self, question, rag_output, retrieved_context):
        metric=FaithfulnessMetric(
                                    threshold=self.threshold,
                                    model=self.evaluation_model,
                                    include_reason=self.include_reason,
                                    verbose_mode=self.verbose_mode
                                 )
        
        test_case=self._test_case(question, rag_output, retrieved_context=retrieved_context)

        return self._run_metric(metric, test_case)
    

    def evaluate_hallucination_metric(self, question, rag_output, retrieved_context):
        metric=FaithfulnessMetric(
                                    threshold=self.threshold,
                                    model=self.evaluation_model,
                                    include_reason=self.include_reason,
                                    verbose_mode=self.verbose_mode
                                 )
        
        test_case=self._test_case(question, rag_output, retrieved_context=retrieved_context)

        result=self._run_metric(metric, test_case)

        # ---- Hallucination score (derived) ----
        result["Score"]=1-result["Score"]
        return result


    def evaluate_all(self, question, rag_output, ground_truth=None, retrieved_context=None):
        results = {}

        # Context-based metrics (require ground truth + retrieval context)
        if ground_truth is not None and retrieved_context is not None:
            results["contextual_precision"] = self.evaluate_contextual_precision_metric(
                question, rag_output, ground_truth, retrieved_context
            )

            results["contextual_recall"] = self.evaluate_contextual_recall_metric(
                question, rag_output, ground_truth, retrieved_context
            )

            results["contextual_relevancy"] = self.evaluate_contextual_relevancy_metric(
                question, rag_output, ground_truth, retrieved_context
            )

        # Answer relevance (does not require ground truth)
        results["answer_relevancy"] = self.evaluate_answer_relevancy_metric(
            question, rag_output
        )

        # Faithfulness (requires retrieved context)
        if retrieved_context is not None:
            results["faithfulness"] = self.evaluate_faithfulness_metric(
                question, rag_output, retrieved_context
            )

            results["hallucination"] = self.evaluate_hallucination_metric(
                question, rag_output, retrieved_context
            )

        return results
