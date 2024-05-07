import { Prop, Schema, SchemaFactory } from '@nestjs/mongoose';
import { HydratedDocument } from 'mongoose';

export type EmployeeDocument = HydratedDocument<Employee>;

@Schema()
export class Employee {
  @Prop()
  name: string;

  @Prop()
  role: string;

  @Prop()
  mailAddress: string;
}

export const EmployeeSchema = SchemaFactory.createForClass(Employee);
