Feature: RAG Chat Bot Query Processing
  As a user
  I want to submit queries to the RAG chat bot
  So that I get answers retrieved from loaded documents

  Scenario: Successful response generation
    Given the RAG bot is initialized with a documents database
    When I submit a query "What is the project description?"
    Then the bot should return a relevant response based on the documents
