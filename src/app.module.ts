import { Module } from '@nestjs/common';
import { SalaryModule } from './salary/salary.module';
import { ConfigModule } from '@nestjs/config';
import { AppController } from './app.controller';
import { MongooseModule } from '@nestjs/mongoose';

@Module({
  imports: [
    SalaryModule,
    ConfigModule.forRoot(),
    MongooseModule.forRoot(process.env.MONGODB_ATLAS_URI),
  ],
  controllers: [AppController],
  providers: [],
})
export class AppModule {}
