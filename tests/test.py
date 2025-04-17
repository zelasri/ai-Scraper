import unittest
import ollama
from langchain_ollama import OllamaEmbeddings

class TestOllamaChat(unittest.TestCase):

    def test_capital_of_france(self):
        # Specify the model name
        model_name = 'llama3.2'

        # Create a client instance
        client = ollama.Client()

        # Send a prompt to the model
        response = client.chat(model=model_name, messages=[
            {'role': 'user', 'content': 'What is the capital of France? respond in 1 word'},
        ])

        # Extract the model's response
        answer = response['message']['content'].strip()

        # Assert that the response is 'Paris', ignoring trailing punctuation or spaces
        self.assertEqual(answer, 'Paris.')

    def test_embed_query(self):
        # Initialize the embedding model
        embeddings = OllamaEmbeddings(model="llama3.2")

        # Input text to embed
        input_text = "The meaning of life is 42"

        # Generate the embedding vector
        vector = embeddings.embed_query(input_text)

        # Define the expected first three values of the embedding vector
        expected_values = [-0.000110041336, 0.01087289, 0.0044805]

        # Assert that each of the first three values is close to the expected value
        for i in range(3):
            self.assertAlmostEqual(vector[i], expected_values[i], places=7)

if __name__ == '__main__':
    unittest.main()
