import { Body, Controller, Get, Post } from '@nestjs/common';
import { SalaryService } from './salary.service';

@Controller('')
export class SalaryController {
  constructor(private salaryService: SalaryService) {}

  @Post('ask')
  async getSalary(@Body('question') question: string) {
    return await this.salaryService.getSalary(question);
  }

  @Get('setup')
  async setupVectorStore() {
    return this.salaryService.setupVectorStore();
  }
}
