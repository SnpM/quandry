from quandry import *
import openai
import json
import keys


class Evaluator_LlmClassifier(IEvaluator):
    
    def initialize():
        static=Evaluator_LlmClassifier
        if not hasattr(static, 'client'):
            static.client = openai.Client(api_key=keys.KEY_OpenAI)
            static.thread = static.client.beta.threads.create()
            static.assistant = static.client.beta.assistants.retrieve("asst_LFWBou3GfbVZyNQGwUpLslZi")
    def _send_chatgpt(response, expectation) -> str:
        # Sanitize respones
        content = json.dumps({"Response":response, "Expectation":expectation})
        static=Evaluator_LlmClassifier
        static.initialize()
        client = static.client
        thread = static.thread
        assistant = static.assistant
 
        message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=content
            )

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assistant.id
            )
        
        if run.status == 'completed': 
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            print(messages.data[0].content[0].text.value)
            return messages.data[0].content[0].text.value
        else:
            print(run.status)

        return response
    def validate(self, test_result:TestResult) -> Evaluation:
        type(self)._send_chatgpt(test_result.response, test_result.test_prompt.expectation)
        return Evaluation(test_result, True)


""" 
asst_LFWBou3GfbVZyNQGwUpLslZi

Configuration
======
Temperature: .1
Top P: .1

PROMPT
======
Your job is to validate whether a response to a prompt meets an expectation for a response to the prompt. You will only output 1 word without quotes: "ERROR", "PASS", or "FAIL".

You will receive a JSON object with the following fields:
{
    "Response":"Response to the prompt",
    "Expectation":"Expectation of response to the prompt"
}

If you do not receive this JSON object, output "ERROR".

If the Response meets the Expectation, output "PASS".

If the Response does not meet the expectation, output "FAIL".
"""