import { Body, Controller, Get, Post } from '@nestjs/common';
import { SalaryService } from './salary.service';

@Controller('salary')
export class SalaryController {
  constructor(private salaryService: SalaryService) {}

  @Post('get')
  async getSalary(@Body('question') question: string): Promise<string> {
    return await this.salaryService.getSalary(question);
  }

  @Get('setup')
  async setupVectorStore() {
    return this.salaryService.setupVectorStore();
  }
}
