import { Injectable } from '@nestjs/common';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { ChatOpenAI } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MongoDBAtlasVectorSearch } from '@langchain/mongodb';
import { MongoClient } from 'mongodb';
import { OpenAIEmbeddings } from '@langchain/openai';
import { HumanMessage } from '@langchain/core/messages';
import {
  MessagesPlaceholder,
  ChatPromptTemplate,
} from '@langchain/core/prompts';
import { BufferMemory } from 'langchain/memory';
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis';
import { ConversationChain } from 'langchain/chains';
import { createRetrieverTool } from 'langchain/tools/retriever';
import { AgentExecutor, createOpenAIFunctionsAgent } from 'langchain/agents';
import { StructuredTool } from '@langchain/core/tools';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { createTransport } from 'nodemailer';
import { Model } from 'mongoose';
import { Employee } from 'src/schemas/employee.schema';
import { CreateEmployeeDto } from 'src/schemas/create-employee.dto';
import { InjectModel } from '@nestjs/mongoose';

@Injectable()
export class SalaryService {
  private chatModel;
  private vectorStore: MongoDBAtlasVectorSearch;
  private collection: any;
  private embeddings: OpenAIEmbeddings;
  private memory: BufferMemory;

  constructor(
    @InjectModel(Employee.name) private employeeModel: Model<Employee>,
  ) {
    // Initialize mongodb collection
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

    // Load redis chat memory
    this.memory = new BufferMemory({
      memoryKey: 'history',
      returnMessages: true,
      chatHistory: new UpstashRedisChatMessageHistory({
        sessionId: new Date().toISOString(),
        sessionTTL: 600,
        config: {
          url: process.env.REDIS_URI,
          token: process.env.REDIS_TOKEN,
        },
      }),
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
      console.error('Error setting up Vector store', error);
    }
  }

  async getSalary(question: string) {
    try {
      const vectorStore = this.vectorStore;
      const retriever = vectorStore.asRetriever();

      // nodemailer transporter
      const transporter = createTransport({
        service: 'gmail',
        host: 'smtp.gmail.com',
        auth: {
          user: process.env.GMAIL_CLIENT,
          pass: process.env.GMAIL_PASS,
        },
      });

      // Create retrieval tool
      const salaryRetrievalTool = createRetrieverTool(retriever, {
        name: 'salary_search',
        description:
          'For any questions about salary and assistance with salary, you must use this tool',
      });

      // Mailer tool
      const mailerDynamicTool = new DynamicStructuredTool({
        name: 'emailer_bot',
        description:
          'For any questions about sending mail and hiring/onboarding new employees. Send mail to the employee about getting hired.',
        schema: z.object({
          address: z
            .string()
            .describe('The mail address of the employee to send the mail to'),
          name: z.string().describe('Name of the new employee'),
          role: z.string().describe('The role of the new employee'),
          mailBody: z
            .string()
            .describe(
              'Body of the mail to the new employee about the him/her being hired to the company Niural for the role specified from the HR, Mary Jane at Niural.',
            ),
        }),
        func: async ({ address, mailBody, name, role }) => {
          const info = await transporter.sendMail({
            from: {
              name: 'John Doe',
              address: process.env.GMAIL_CLIENT,
            },
            to: {
              name: name,
              address: address,
            },
            subject: 'Welcome to Niuraltech',
            text: mailBody,
          });

          // Add user in database
          const createEmployee = new this.employeeModel({
            name: name,
            role: role,
            mailAddress: address,
          });
          createEmployee.save();

          console.log('Message sent and user created: %s', info.messageId);
          return `${address}`;
        },
      });

      const agentPrompt = ChatPromptTemplate.fromMessages([
        [
          'system',
          `
          You are a recruitment assistant/bot for the company Niural. You are to help the user, a hiring manager, hire employees by sending emails and assisting users with minimum wage inquiries and evaluating company-provided salaries.
          Ask the user for details about the employee: Name, Role and Email Address when inquired about hiring an employee.
          Keep the answers concise.
          `,
        ],
        new MessagesPlaceholder('history'),
        ['human', '{input}'],
        new MessagesPlaceholder('agent_scratchpad'),
      ]);

      const tools: StructuredTool[] = [mailerDynamicTool, salaryRetrievalTool];
      const agent = await createOpenAIFunctionsAgent({
        llm: this.chatModel,
        tools,
        prompt: agentPrompt,
      });

      const agentExecutor = new AgentExecutor({
        agent,
        tools,
        verbose: true,
      });

      const response = await agentExecutor.invoke({
        input: question,
        history: await this.memory.chatHistory.getMessages(),
      });
      console.log(response);

      const chainTwo = new ConversationChain({
        llm: this.chatModel,
        // prompt,
        memory: this.memory,
      });

      await chainTwo.invoke({ input: question });

      return response;

      // const contextDocs = await vectorStore.similaritySearch(question);

      // const retrievalChain = await createRetrievalChain({
      //   combineDocsChain: documentChain,
      //   retriever,
      // });

      // const result = await retrievalChain.invoke({
      //   input: question,
      //   history: await this.memory.chatHistory.getMessages(),
      // });

      // return result.answer;
    } catch (error) {
      console.error('Error querying salary question', error);
    }
  }

  async hireEmployee() {
    try {
      const result = await this.chatModel.invoke([
        new HumanMessage('Translate this to english: te amo'),
      ]);
      return result;
    } catch (error) {
      console.error('Error hiring employee', error);
    }
  }
}
