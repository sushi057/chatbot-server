import { Module } from '@nestjs/common';
import { SalaryModule } from './salary/salary.module';
import { ConfigModule } from '@nestjs/config';

@Module({
  imports: [SalaryModule, ConfigModule.forRoot()],
  controllers: [],
  providers: [],
})
export class AppModule {}
