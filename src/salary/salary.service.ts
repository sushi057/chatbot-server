import { Injectable } from '@nestjs/common';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { ChatOpenAI } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MongoDBAtlasVectorSearch } from '@langchain/mongodb';
import { MongoClient } from 'mongodb';
import { OpenAIEmbeddings } from '@langchain/openai';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';
import { MessagesPlaceholder } from '@langchain/core/prompts';

@Injectable()
export class SalaryService {
  private chatModel;
  private vectorStore: MongoDBAtlasVectorSearch;
  private collection: any;
  private embeddings: OpenAIEmbeddings;

  constructor() {
    // Initialise mongodb collection
    const client = new MongoClient(process.env.MONGODB_ATLAS_URI || '');
    const namespace = 'niuraltech.chatbot';
    const [dbName, collectionName] = namespace.split('.');
    this.collection = client.db(dbName).collection(collectionName);

    // Load openAI Embeddings
    this.embeddings = new OpenAIEmbeddings({
      dimensions: 1536,
      modelName: 'text-embedding-3-large',
      azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
      azureOpenAIApiVersion: process.env.AZURE_OPENAI_API_VERSION,
      azureOpenAIApiInstanceName: process.env.AZURE_OPENAI_API_INSTANCE_NAME,
      azureOpenAIApiDeploymentName:
        process.env.AZURE_OPENAI_API_EMBEDDINGS_DEPLOYMENT_NAME,
      azureOpenAIBasePath: process.env.AZURE_OPENAI_BASE_PATH,
    });

    this.vectorStore = new MongoDBAtlasVectorSearch(this.embeddings, {
      collection: this.collection,
      indexName: 'vector_index',
      textKey: 'text',
      embeddingKey: 'embeddings',
    });

    this.chatModel = new ChatOpenAI({
      temperature: 0.5,
      azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY,
      azureOpenAIApiVersion: '2024-02-15-preview',
      azureOpenAIApiInstanceName: 'super-open-ai-east',
      azureOpenAIApiDeploymentName: 'superai-35-turbo-16k',
      azureOpenAIBasePath:
        'https://super-open-ai-east.openai.azure.com/openai/deployments/superai-35-turbo-16k/chat/completions?api-version=2024-02-15-preview',
    });
  }

  async setupVectorStore() {
    try {
      // Clear collection
      await this.collection.deleteMany({});

      // Load website (US State Labor Laws)
      const webLoader = new CheerioWebBaseLoader(
        'https://www.dol.gov/agencies/whd/minimum-wage/state',
      );
      const docs = await webLoader.load();

      const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 3000,
        chunkOverlap: 500,
      });
      const splitDocs = await splitter.splitDocuments(docs);
      console.log(splitDocs);

      const embeddings = this.embeddings;

      const vectorStore = await MongoDBAtlasVectorSearch.fromDocuments(
        splitDocs,
        embeddings,
        {
          collection: this.collection,
          indexName: 'vector_index',
          textKey: 'text',
          embeddingKey: 'embeddings',
        },
      );

      this.vectorStore = vectorStore;

      console.log(
        `Loaded mongodb vector store with ${splitDocs.length} documents`,
      );

      return { success: 'MongoDB Vector Store setup' };
    } catch (error) {
      console.error('Error seting up vectorstore', error);
    }
  }

  async getSalary(question: string) {
    try {
      const prompt = ChatPromptTemplate.fromTemplate(
        `
        "You are a chatbot specializing in assisting users with minimum wage inquiries and evaluating company-provided salaries. You provide analytical responses to questions, guiding users in making informed decisions regarding their wages
        Keep the answers concise.

        <context>
        {context}
        </context>

        Question: {input}

        `,
      );

      const documentChain = await createStuffDocumentsChain({
        llm: this.chatModel,
        prompt,
      });

      const vectorStore = this.vectorStore;
      const retriever = vectorStore.asRetriever();

      const retrievalChain = await createRetrievalChain({
        combineDocsChain: documentChain,
        retriever,
      });

      const historyAwarePrompt = ChatPromptTemplate.fromMessages([
        [
          'system',
          `
          "You are a chatbot specializing in assisting users with minimum wage inquiries and evaluating company-provided salaries. You provide analytical responses to questions, guiding users in making informed decisions regarding their wages.
          Answer questions based on the given context.

          <context>
          {context}
          </context>
          `,
        ],
        new MessagesPlaceholder('chat_history'),
        ['human', '{input}'],
      ]);

      const historyAwareCombineDocsChain = await createStuffDocumentsChain({
        llm: this.chatModel,
        prompt: historyAwarePrompt,
      });

      const historyAwareRetrieverChain = await createHistoryAwareRetriever({
        llm: this.chatModel,
        retriever,
        rephrasePrompt: historyAwarePrompt,
      });

      const conversationalRetrievalChain = await createRetrievalChain({
        retriever: historyAwareRetrieverChain,
        combineDocsChain: historyAwareCombineDocsChain,
      });

      const result2 = await conversationalRetrievalChain.invoke({
        input: question,
      });

      const contextDocs = await vectorStore.similaritySearch('colorado?');

      const result = await retrievalChain.invoke({
        input: question,
      });

      return result.answer;
    } catch (error) {
      console.error('Error querying salary question', error);
    }
  }
}
