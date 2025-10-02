import re
import numpy as np
from scipy.spatial.distance import cosine
import google.generativeai as genai
import logging
import pickle
import os
import time
import sys
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type

# --- Configuration & Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_ACTUAL_GEMINI_API_KEY")  # Use environment variable
if GEMINI_API_KEY == "YOUR_ACTUAL_GEMINI_API_KEY" or not GEMINI_API_KEY:
    logging.error("CRITICAL ERROR: Please set GEMINI_API_KEY environment variable or replace 'YOUR_ACTUAL_GEMINI_API_KEY' with a valid key.")
    print("CRITICAL ERROR: Please set GEMINI_API_KEY environment variable or replace 'YOUR_ACTUAL_GEMINI_API_KEY' with a valid key.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EMBEDDINGS_CACHE_FILE = "gemini_embeddings_cache.pkl"
EMBEDDING_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.0-flash"
MAX_TOKENS_FOR_EMBEDDING = 4000

# --- Manual Text (Replace with your full manual content) ---
manual_text = """
INTRODUCTION
Overview of VedCool Platform
VedCool is a cutting-edge educational platform designed to simplify and enhance the management of
academic and administrative processes. By integrating advanced technology with user-friendly features,
VedCool caters to the needs of diverse stakeholders, including administrators, teachers, students, and parents.
The platform offers three core solutions:
• VedCool Stream: An engaging and interactive module for streaming educational content, fostering
immersive learning experiences.
• VedCool Campus: A comprehensive system for managing campus-wide operations, including
scheduling, resource allocation, and communication.
• VedCool Learn: A dedicated learning platform that personalizes education for students, empowering
them to excel academically.
VedCool ensures seamless integration and adaptability, making it an essential tool for modern educational
institutions.
Overview of the Branch Admin Role
The Branch Admin role is designed to streamline the management of individual branches within the
organization. As a Branch Admin, users are granted specific permissions to oversee branch-level operations,
ensuring efficient functioning and adherence to organizational standards. This role bridges the gap between
branch-level staff and higher-level administration, enabling a smooth flow of information and operations.
Key functionalities of the Branch Admin role include:
• Managing branch-specific data, including employee records and customer details.
• Monitoring branch performance metrics and generating reports.
• Addressing operational issues and implementing solutions promptly.
• Acting as the primary point of contact for branch-level queries and escalations.

  User Manual
5

Key Responsibilities
Branch Admins play a crucial role in maintaining the operational and administrative health of their respective
branches. Below are the core responsibilities associated with this role:
Data Management
• Maintain accurate records of branch employees, clients, and inventory (if applicable).
• Update and verify branch-related data in the system regularly.
Operational Oversight
• Monitor branch-level operations to ensure they align with organizational policies.
• Approve or escalate requests related to resource allocation, scheduling, or procurement.
Reporting and Analytics
• Generate periodic reports to evaluate branch performance.
• Use system analytics to identify areas of improvement and implement necessary actions.
User Management
• Assign roles and permissions to branch-level staff within the system.
• Reset passwords and resolve basic user access issues.
Communication and Support
• Act as the liaison between branch staff and corporate management.
• Address escalations promptly and provide guidance to branch employees.

BRANCH ADMIN
The Branch Admin role in VedCool manages branch-level operations, including admissions, student and
employee records, academics, and finances. They ensure seamless administration through reporting,
scheduling, and communication tools.
Login
Access the Login Page
• Open your preferred web browser, such as Google Chrome, Firefox, or Safari.
• In the address bar at the top, type www.vedcool.ai and press Enter.
• The VedCool website will appear

  User Manual
6

• Click on the Log In, to access to the page further
• The Log In page will appear infront of you, enter the credentials to access the page
• Enter your Email and Password in the respective fields.
• Click the Login button to proceed.
Setting Up Profile
Log In to Your Account
• Use your credentials to log in to the VedCool platform.
Access your Profile
• Click on your profile icon or username in the top-right corner of the dashboard.
  User Manual
7

• From the dropdown menu, select Profile.
• This profile page will be displayed to you.
Update Profile & Change Password
• Edit Your Information
• In the profile settings page, locate the Profile section.
• Update fields such as:
o Name
o Email Address
o Phone Number
o Address
• Ensure all details are accurate and up-to-date.
• Upload or Update Profile Picture
• Click on the profile picture placeholder or current image.
• Select an image from your device and upload it.
• Save Changes
• Once all changes are made, click the Save or Update button to confirm.
  User Manual
8

Changing Passwords
• Click on your profile icon.

• Select Reset Password from the dropdown menu.
• In the Current Password section enter your current password.
• Enter a new password in the New Password field.
• Re-enter the new password in the Confirm Password field.
• Ensure the new password meets the security criteria:
  User Manual
9


• Save Changes
• Click the Update button, to save the changes you have made.
• A confirmation message will appear indicating the password has been successfully changed.
DASHBOARD
Upon successful login, you will be redirected to the Branch Admin dashboard, where you can manage
branches, users, and global settings.


  User Manual
10

Academic Insights
Provides an overview of the academic performance, attendance, and curriculum management within the
institution.
Key Features:
• Student Academic Performance Analytics Overview: Provides a comprehensive analysis of student
performance across subjects, helping educators and administrators identify trends, strengths, and areas
for improvement.
• Student Enrollment Analytics: Offers insights into student enrollment trends, including
demographics, admission rates, and retention patterns to inform strategic decision-making.
• Daily Attendance: Tracks and reports daily student attendance, helping schools monitor attendance
patterns and manage absenteeism.
• Student Progress Analytics: Evaluates student development over time, highlighting progress in
academic achievement, skill acquisition, and overall growth.
• Homework Analytics: Analyzes homework completion rates and performance, providing feedback
on student engagement and areas requiring additional support.
• Top Performance: Highlights top-performing students based on grades, behavior, or other criteria,
recognizing academic excellence and motivating other students.
Operational Insights
It involves analyzing key performance data across various organizational functions to optimize processes,
improve efficiency, and support data-driven decision-making. It helps track performance, resource allocation,
and operational effectiveness in real time.
  User Manual
11


Key Features:
• Employee Attendance: Provides a detailed overview of employee attendance, including punctuality,
absenteeism, and leave trends, helping HR and management ensure workforce efficiency.
• Payroll Summary: Summarizes employee payroll data, including salary disbursements, bonuses,
deductions, and benefits, offering transparency and accuracy in financial management.
• Bus Tracking: Monitors the location and route of school buses in real time, ensuring safe and timely
transportation for students, while also optimizing routes for efficiency.
• Classroom Utilization: Analyzes classroom usage, including occupancy rates and scheduling
efficiency, to optimize space allocation and improve the learning environment.
• Budget Utilization: Tracks the allocation and spending of the organization's budget, offering insights
into financial health and helping with cost management and planning.
• Library Analytics: Tracks library usage, including book checkouts, student engagement, and popular
resources, helping improve library services and manage inventory effectively.
Financial Overview & Events
Financial Overview Provides a summary of the organization's financial health, including income, expenses,
budget utilization, and outstanding dues, enabling effective financial planning and management.
Events Tracks and manages organizational events, including schedules, participation, and budgets, ensuring
seamless execution and improved engagement.
  User Manual
12


Key Features:
• Fee Collection: Tracks and manages all fee payments, including pending and completed transactions,
ensuring timely and efficient collection.
• Fee Summary: Provides a consolidated view of fee-related data, including collections, outstanding
balances, and categorized revenue reports.
• Upcoming Event: Displays scheduled events with details such as date, time, and venue, helping
stakeholders stay informed and prepared.
• Announcement: Shares important updates and notifications with students, parents, and staff, ensuring
effective communication within the organization.

ADMISSION
Admission Management in VedCool allows Branch Admins to efficiently handle student admissions, view
and edit admission records, and generate comprehensive reports. Here’s a step-by-step guide for each of the
functions under Admission Management
Create Admission
• Navigate to the Admission section from the main menu.
• Click on ‘Create Admission’
• In the Admission section, click the ‘Create Admission’ button
  User Manual
13

This will open the admission form to enter new student details.
• Enter the required student information, including:
o Name
o Date of Birth
o Contact Information
o Previous Academic Details (if applicable)
o Assigned Class/Grade
o Parent/Guardian Details
o Admission Date
o Ensure all fields marked with * are filled out
• Use the ‘Upload’ button to select and upload files or drag and drop option.
• Once all details are filled in, click the ‘Save button.
  User Manual
14

A confirmation message will appear to notify you that the student’s admission has been added successfully.
Multiple Import
• Navigate to the Admission section from the main menu.
• Click on ‘Multiple Import’
• For importing multiple data, you must have the data in ‘csv’ format
• Download the csv file, and choose the standard and section from the dropdown, for which you want to
enter the records
• Then upload your csv data set file
• Click on Import to proceed.
  User Manual
15

Category
• From the category section, the branch admin can create the category as per the requirement.
• Simply click on the admission, and then navigate to category
• Enter the branch name, and click on save button to save the branch.

STUDENT DETAILS
After logging in, follow these steps to access the Student Details section, which includes the Student List,
Inquiry List, ID Cards, and the ability to Deactivate Student Login.
Student List
• Go to the Student Section
• Open the Dashboard.
• From the menu, find and click on Student Details.
• In the dropdown, select Student List.
• Select the Academic Session
  User Manual
16

• Use the Academic Session dropdown at the top-right corner.
• Pick the desired session to view the student list.
View the Student Details
The list of all enrolled students will appear, showing details such as:
• Student Photo
• Roll Number
• Student Name
• Age
• Gender
• Phone Number
• Caste
• Registration Number
• Standard/Section
• Father’s Name
• Use the Standard dropdown to see students of a specific class.
• Select a Section from the dropdown to refine the list further.
Search and Filter
• By clicking on the filter button, whatever you have selected above it will showcase the data on that
basis.
View/Edit Student Details

  User Manual
17

• To view the quick details of the student, on the right side in the Action column click the Quick View
Button
• This will showcase the quick summary of the student.
• To view the detailed summary of the student, on the right side in the Action column click the Details
Button

  User Manual
18

• This will showcase the detailed summary of the student.
• To delete the record of the particular student, on the Action column click the button Delete
• From here, you can Delete single details about the student.
• But to delete all the data, you can use Bulk Delete
• This will delete the complete details of the student

Inquiry List
Access the Inquiry List
• From the Vedool Dashboard, navigate to the Student Details section.
  User Manual
19

• Click on Inquiry List to view all student inquiries.
Filter by Standard and Section
• Use the Standard dropdown to select the class you want.
• Choose the Section from the dropdown to narrow down the inquiries.
Manage the Inquiry List:
• In the Inquiry List panel, you can:
• Copy the list for quick reference.
• Download it as an Excel or CSV file.
• Print the list for offline use.
• Customize the displayed data by selecting the columns you want to view.
Delete Records in Bulk
• To delete multiple inquiries at once, click on the Bulk Delete button.
  User Manual
20

• Confirm your action to remove all selected records in a single step.
ID Card
Click on 'ID Card'
• Under the Student Details section, click on the ID Card option.
Generate or Print ID Cards
• From here, you can generate ID Cards for students.
• Select the Standard & Section, to view the specific class / student data
• After selecting click on the filter button to apply the filter and view the result.
• You will have the option to select a student and click on Checkbox of the student whose ID Card you
want to Print.
• After selecting the checkbox, just simply click on the print button and then you can print the ID Card
of that particular student.

Login Deactivate
Access the Login Deactivate Section
• From the Vedool Dashboard, go to the Student Details section.
• Click on Login Deactivate to manage student login statuses.
Filter by Standard and Section
• Use the Standard dropdown to select the class.
• Choose the Section from the dropdown.
• Click on the Filter button to apply the filters and display the student list.
  User Manual
21

Deactivate a Student’s Login
• The list of students with login credentials will appear.
• Select the student(s) whose login you want to deactivate.
• Click the Deactivate button.
Confirm Deactivation
• A confirmation pop-up will appear.
• Click Yes to confirm and deactivate the login.
VEDCOOL LEARN
VedCool Learn is the central learning management platform within VedCool that offers a variety of educational
tools and resources to enhance the learning experience for students, educators, and administrators. The
platform is designed to streamline the teaching and learning process by providing easy access to courses, study
materials, assessments, and progress tracking.
• Accessing Learning Resources
• From the category section, the branch admin can create the category as per the requirement.
• Simply click on the admission, and then navigate to category
• Enter the branch name, and click on save button to save the branch.
• Once logged in and on the Dashboard, find and click on the VedCool Learn option in the left-side
navigation menu or under the Learning Management section
  User Manual
22

• Clicking on VedCool Learn will open the learning platform interface where you can access various
learning resources, courses, and materials.
• Click on the Log in button to login into the LMS portal of vedcool.
• Enter the credentials and login to the LMS portal.
• If you are new to the LMS portal then click on the Sign up
• Then this page navigates you to the registration form which looks like:
• Fill the details and then click on Sign Up and you’ll have the successful sign up to the LMS portal.
After the login the page will look like:
• Under the profile section you can access these many features:
  User Manual
23

• Know to view my courses, there are two ways:
• One is to click the My Courses from the navbar and you can see your all the purchases / free courses.
• Second, by clicking on my profile from there you can access My Courses
• This will navigate you to the My Courses which looks likes:
Create Instructor
• If the student or some other persons wants to become the instructor then by navigating the instructor
from navbar the one can do so.
• From the home page click on the instructor
• This will redirect you to the form, by filling which the one can enrols them self to be an instructor
• Fill the necessary details, and then click on Apply.
• You profile will be created as an instructor
  User Manual
24

Navigate the Interface
• From the My Messages option we can compose the message for the students.
• From the purchase history, the one can view what he/she has purchased, here they’ll get the complete
details about the purchase item.
• In the User Profile section user can update his/her profile as per her need
• We can update the profile photo
• We can add biography which can describe our self in a short and beautiful manner so that any one
viewing our profile and know about ourself.
  User Manual
25


VIRTUAL CLASS
The Virtual Class module in VedCool enables seamless online learning through live video sessions. Teachers
can schedule, conduct, and manage virtual classes with interactive features like screen sharing, chat, and
attendance tracking. Students can join classes, access recorded sessions, and participate in real-time
discussions.
Setting up Virtual Class
• Once logged in, from the Dashboard, look for the Virtual Class or Live Class section.
• Click on Virtual Class to access the virtual class management interface.
Host Meating
• To host the meeting, simply click on the Host Meating
• Enter the meeting title, and generate the password
  User Manual
26

• And then click on Create & Join Now
Joining Class
• To join the meeting, simply enter the credentials and click on the Join Now button, to join the meeting.
PARENTS
The Parents module in VedCool allows guardians to monitor their child's academic progress, attendance, and
fee status. Parents can access student reports, receive notifications about exams and events, and communicate
with teachers. This module ensures transparency and active parental involvement in the child’s education.
Parent List
• After logging in, locate the Parents section in the navigation menu or dashboard sidebar.
  User Manual
27

• Click on Parents to open the Parent Management page, where you can manage parent-related data.
• Once you are in the Parents section, click on the Parent List tab to view all registered parents.
• The list will display details like Guardian Name, Student Name, Occupation, Mobile No, Email.
• the search bar or filter options to find a specific parent or group of parents based on criteria like name
or student name.
• Select the standard and section and click on filter button to apply the filter to View Parent Details
• To view the complete profile details of any parent and their children, click on the Profile button next
to their name, in the action column.
• This will open the parent’s full profile, showing all their personal and contact information.
• Here you can also view the basic details of the parents, and can also update them if needed.
• After making the changes click on the update button to save the changes.
  User Manual
28

• To view the child details, click on the Childs to view the children’s complete details.
• To view the children’s profile, simply click on the Profile button, and this will navigate you to
Students Profile.
• Where you can find students complete details.
• To Delete any record of the parent you can click on Delete button in the action column.
  User Manual
29

Add Parent
• To add the parents, click on Add Parent
• From the Parent, click on the dropdown and locate and click the Add Parent button to add a new
parent to the system.
• This will navigate you to Add Parent page
Enter Parent Information
• Fill in the necessary details for the new parent, such as:
o Parent Name: Full name of the parent.
o Relationship to Student: Specify the relationship (e.g., Father, Mother, Guardian).
o Email Address: Parent’s email for communication.
o Phone Number: Contact number of the parent.
o Address: Parent’s residential address.
o Student Information: Link the parent to a specific student by entering the student's name or ID.
Save the New Parent Record
• After entering the details, click on the Save button to add the parent to the system.
  User Manual
30


Login Deactivate
  Access the Parent List
• From the Parent, click on the dropdown and locate and click the Login Deactivate button to deactivate
the parent.
Select the Parent to Deactivate
• Locate the parent whose login you want to deactivate.
• Click on the Deactivate button next to their name in the action column.
• From the panel you can:
• copy the Inquiry list
• download excel format
• download csv format
• print the list
  User Manual
31

• and can view as per the columns
Confirm Deactivation
• A confirmation prompt will appear asking if you’re sure you want to deactivate the parent’s login.
• Confirm the action by clicking Yes or Deactivate.
• Successful Deactivation
• The parent’s login will be deactivated, and they will no longer be able to log in to the system using
their credentials.
• A confirmation message will notify you that the parent’s login has been successfully deactivated.

EMPLOYEE
The Employee module in VedCool manages staff details, including personal information, department,
designation, and login credentials. Administrators can track attendance, payroll, leave records, and
performance reports. This module ensures efficient HR management and streamlines employee-related
operations.
Add Department
• To add the employee department, from the Dashboard page select Employee
• Then from the drop down select the Add Department
• Navigate to Employee in the main menu.
• Click on Departments Name, and enter the name of the Department
• And click on Save button to save the action.
• To edit the particular department, from the action column click on edit button

  User Manual
32

This will lead you to edit you department name, make the changes and click on the update button.
• To delete the particular department name, click on delete button from the action column.
• This will Delete that particular department for you.
Add Designation
• To add the employee designation, from the Dashboard page select Employee
• Then from the drop down select the Add Designation
• Navigate to Employee in the main menu.
• Click on Designation Name, and enter the name of the Designation



  User Manual
33

• And click on Save button to save the action
• To edit the particular designation, from the action column click on edit button
• This will lead you to edit you designation name, make the changes and click on the update button.
• To delete the particular designation name, click on delete button from the action column.
• This will Delete that particular department for you.
  User Manual
34


Add Employee
• To add the employee, from the Dashboard page select Employee
• Then from the drop down select the Add Employee
• Navigate to Employee in the main menu.
• Click on Add Employee.
• Fill in the details (e.g., Name, Contact, Designation).
• Upload documents or profile images as required.
• Click Submit.

  User Manual
35


Employee List
• To view all the employee, from the Dashboard page select Employee
• Then from the drop down select the Employee list
• This will redirect you to the page where you ca see the list of all the Employee
• The list will show all the category of the employee ‘Admin’, Teacher’, ’Accountant’, ’Librarian’,
’Receptionist’, ’Peon’.


  User Manual
36

Login Deactivate
• Go to Employee > Login Deactivate
• Select the employee whose login you want to deactivate.
• Click on Deactivate Login.
• Confirm the action.

HR (HUMAN RESOURCE)
The HR module in VedCool handles employee management, including payroll processing, leave tracking, and
certifications. Administrators can manage salary structures, approve leave requests, and maintain employee
records. This module ensures smooth human resource operations and compliance with organizational policies.
Pyroll
• Navigate to the Salary Template
• From the Dashboard, go to the Human Resource section.
• Select Payroll and click on Salary Template.
  User Manual
37

Create/Edit a Salary Template
• Click on the Create Template button to add a new template.
• Fill in the required fields such as template name, basic salary, allowances, deductions, and tax details.

  User Manual
38

• Save the template for future use.
• To edit an existing template, select it from the list, make changes, and save.
Advance Salary
• Go to HR > Advance Salary.
• Select the advance salary request and enter the advance amount and reason,
• Click Submit.

  User Manual
39

Leave
• Navigate to HR > Leave Management.
• View leave requests or apply for new leaves.
• If you need to your leave application, then navigate to My Application.
• Approve or reject leaves as necessary.





  User Manual
40

Certification
• Go to HR > Certification.
• Add or view employee certifications.

ACADEMIC
The Academic module in VedCool manages the school's curriculum structure, including standards, sections,
subjects, and timetables. Administrators can configure academic sessions, promote students, and assign
subjects to different grades. This module ensures efficient academic planning and organization.

Standard Section
• Navigate to Academic > Standard Section.
• Add new standards or sections by clicking Add New.
• Enter the required details and click Save.
  User Manual
41

• Add new teacher by clicking Assign New Teacher
Subject
• Go to Academic > Subject.
• Add, manage or assign subjects for specific standards.
• Click Save after entering details.
• View the Teacher Assign List, you can copy, save or print the list also.

Timetable
• Navigate to Academic > Timetable.
• Set up timetables for classes and sections.
• Save the timetable.
Promotion
• Go to Academic > Promotion.
• Select the standard and students to be promoted.
• Click Promation.

LIVE CLASSROOMS
The Live Classroom module in VedCool facilitates real-time online teaching with interactive features like
video conferencing, screen sharing, and live chat. Teachers can schedule and conduct virtual sessions, while
students can join, participate, and access recorded lectures. This module enhances remote learning and
engagement.
• Setting up Live Classroom
• Navigate to Live Classrooms.


• Configure the settings for live classrooms, such as linking a platform.
• Click Save.
• Managing Live Class Session
• Go to Live Classrooms > Add Live Class.
• Create the live class session and click on save button to save.

ATTACHMENTS / BOOKS
The Books module in VedCool manages the library system, including book categories, availability, and
issuance records. Librarians can add, update, and track books, while students and staff can check availability
and request book issues. This module ensures efficient library management and seamless book tracking.
Attachment Type
• Navigate to Attachments/Books > Attachment Type.


• Add or edit attachment types.
Upload Content
• Go to Attachments/Books > Upload Content.


• Upload documents or study materials.
• Click Submit.

HOMEWORK
The Homework module in VedCool allows teachers to assign, track, and evaluate student assignments online.
Students can submit their homework digitally, and teachers can review, grade, and provide feedback. This
module enhances learning engagement and streamlines homework management.
Assigning Homework
• Navigate to Homework > Assign Homework.
• Select the class, subject, and description.
• Click Assign.
Homework Evaluation Report
• Go to Homework > Evaluation Report.

• View or download the reports.

EXAM MASTER
The Exam Master module in VedCool enables administrators to create, schedule, and manage exams
efficiently. Users can define exam names, set dates, assign classes, and configure grading criteria. This module
ensures streamlined exam planning, execution, and result management.
Exam Setup
• Navigate to Exam Master > Setup Exam.
• Add Exam term details.
• Add details like exam name, date, and classes involved.
• Add Exam Hall Details, add the necessary details and then you can view the Exam Hall List.
• Click Save.
• Add the distribution details by licking the Distribution
• Create a Exam by clicking Exam Setup

Exam Timetable
• Go to Exam Master > Exam Timetable.
• Apply the filter to search accordingly.
• Create or edit exam schedules.
• Save the changes.
Marks
• Navigate to Exam Master > Marks Entry.
• Enter marks for exams and click Save.
• Create the Grade, by filling the necessary details and then click on save button.

SUPERVISION
The Supervision module in VedCool oversees various administrative aspects, including hostel management,
student discipline, and facility monitoring. It helps administrators track student activities, allocate hostel
rooms, and generate supervision reports. This module ensures a well-organized and secure school
environment.
Managing Hostel
• Navigate to Supervision > Hostel Management.
• Click on Hostel to access hostel management features.
• Navigate to Supervision > Hostel > Category.
• Click on Add New Category.
• Enter details such as:
• Category Name (e.g., Boys Hostel, Girls Hostel, Staff Quarters).
• Hostel Type (Day Boarding, Residential).


• Facilities provided in the hostel.
•  Click Save to store the category details.
• Add or manage hostel facilities.
• Go to Supervision > Hostel > Hostel Master.
• Click Add New Hostel.
• Enter the following details:
• Hostel Name (e.g., VedCool Residential Block A).
• Hostel Category (select from previously created categories).
• Total Capacity (number of students that can be accommodated).
• Warden Details (name and contact of the warden).
• Facilities Available (Wi-Fi, Mess, Laundry, etc.)
• Click Save to finalize the hostel master setup.
• Navigate to Supervision > Hostel > Hostel Room.
• Click Add New Room.


• Enter room details such as:
o Room Number/Name.
o Hostel Name (select from existing hostels).
o Room Type (Single, Double, Dormitory, etc.).
o Bed Capacity (total number of students per room).
o Available Seats (seats available for allocation).
o Click Save to store the room details.
o Navigate to Supervision > Hostel > Allocation Report.
o View the list of students assigned to hostel rooms.
o Filter the report based on hostel name, room number, or student name.



  User Manual
53


• Export the report if needed for record-keeping.
Transport
• Navigate to Transport > Vehicle Master
• From the dashboard, go to the Supervision module.
• Click on Vehicle Master to manage the vehicles.
• Click Add New Vehicle.
• Fill in the vehicle details:
o Vehicle Number (e.g., ABC1234).
o Vehicle Type (Bus, Van, etc.).
o Driver Name and Contact Information.
o Capacity (number of seats).
o Route Number (if applicable)
  User Manual
54


• Click Save to add the vehicle to the system.
• Go to Transport > Stoppage.
• Click Add New Stoppage.

• Enter the stoppage details:
• Stoppage Name (e.g., Main Street, Central Park).
• Location Details (landmarks, address).
• Arrival Time (estimated time of vehicle arrival).

• Click Save to store the stoppage information.
• Navigate to Transport > Trip.
• Click Add New Trip.
  User Manual
55


• Provide trip details:
o Trip Name (e.g., Morning Route 1).
o Vehicle Number (select from Vehicle Master).
o Driver Name (auto-filled based on vehicle selection).
o Route (select the sequence of stoppages created earlier).
o Timings (start and end time for the trip).

• Click Save to finalize the trip setup.

ATTENDANCE
The Attendance module tracks student, employee, and exam attendance. It allows teachers to mark daily
attendance for students and employees, while administrators can view and generate attendance reports for
both.
Student Attendance
• Navigate to Attendance > Student Attendance.
• Mark attendance for students.
  User Manual
56


Employee Attendance
• Go to Attendance > Employee Attendance.
• Manage attendance records for employees.

Exam Attendance
• Navigate to Attendance > Exam Attendance.
• Mark attendance for students during exams.


LIBRARY
The Library module manages books, categorizes them, and tracks issued/returned books. Students and staff
can check out books, and librarians can monitor availability and due dates. It helps streamline the library's
operations.
Books Category
• Navigate to Library > Books Category.
• Add new categories for books.
  User Manual
57


Books Management
• Go to Library > Books Management.
• Add or edit book details.

My Issued Books
• Navigate to Library > My Issued Books.

• View the list of issued books.
  User Manual
58


Book Issue / Return
• Go to Library > Book Issue/Return.

• Manage book transactions.

EVENT
The Events module enables administrators to create, manage, and track school events. It includes event types,
scheduling, and participant management, ensuring smooth event organization.
Event Type
• Navigate to Event > Event Type.
• Add or edit event types.
  User Manual
59


Creating & Managing Events
• Go to Event > Manage Events.
• Add new events and manage existing ones.


BULK SMS AND EMAIL
The Bulk SMS and Email module allows administrators to send notifications to students, parents, and staff
in bulk. This module includes SMS/email templates and reports to track delivery and responses.
Sending SMS / Email
• Navigate to Bulk SMS and Email > Send.
• Compose and send messages.
  User Manual
60


• Compose and send email

SMS / Email Report
• Go to Bulk SMS and Email > Reports.
• View sent message logs.
  User Manual
61


SMS Template
• Navigate to Bulk SMS and Email > SMS Templates.
• Add or edit templates for SMS.

Email Templates
• Go to Bulk SMS and Email > Email Templates.
• Add or edit templates for emails.

  User Manual
62

STUDENT ACCOUNTING
The Student Accounting module manages fees, payments, and invoices. It tracks fee types, schedules,
payments, and generates due fee reminders, ensuring financial transparency.
Fee Type / Group
• Navigate to Student Accounting > Fee Type/Group


  User Manual
63

• Add or edit fee types.
Fine Setup
• Go to Student Accounting > Fine Setup.
• Add or edit fine rules.
Fee Allocation
• Navigate to Student Accounting > Fee Allocation.
• Assign fees to students.
  User Manual
64


Fee Payment / Invoice
• Go to Student Accounting > Payment/Invoice.
• Process fee payments.

Due Fees Invoice
• Navigate to Student Accounting > Due Fees.
• View and manage unpaid fees.



  User Manual
65

Fee Reminder
• Go to Student Accounting > Fee Reminder.
• Send reminders for unpaid fees.

OFFICE ACCOUNTING
The Office Accounting module tracks all school financial transactions, including deposits, expenses, and
account vouchers. It provides insights into financial health through real-time reports.
  User Manual
66

Account Management
• Navigate to Office Accounting > Accounts.
• Add or manage accounts.

Voucher Head
• Go to Office Accounting > Voucher Head.
  User Manual
67

• Add or edit voucher heads.
New Deposit
• Navigate to Office Accounting > New Deposit.
• Record deposit details.
  User Manual
68

New Expense
• Go to Office Accounting > New Expense.
• Record expense details.





  User Manual
69

All Transaction
• Navigate to Office Accounting > Transactions.
• View transaction history.


MESSAGE
The Message module facilitates internal communication within VedCool. Users can send messages to
individuals or groups, ensuring smooth collaboration between students, teachers, and staff.
Sending Messages
• Navigate to Message > Compose.
• Send messages to users or groups.
  User Manual
70


REPORTS
The Reports module offers various pre-configured reports such as attendance, financial, academic, and exam
related reports. It enables administrators and teachers to analyze and export reports for better decision-making.
Fee Reports
• Navigate to Reports > Fees Reports.
• From the dashboard, go to Reports.
• Click on Fees Report.
• Select Academic Year.
• Choose the Class & Section (or all classes).
• Select a Fee Category (Tuition Fee, Transport Fee, Hostel Fee, etc.).

• Click Generate Report to view total collected fees.
• The report shows Total Fees, Paid Amount, and Balance per student.
• Use filters to refine the data based on payment status.
  User Manual
71


• Click Export (CSV/Excel/PDF) for record-keeping.
Receipts Report
• Navigate to Reports > Receipts Report
• From the dashboard, go to Reports.
• Click on Receipts Report.
• Select Date Range (e.g., last 7 days, last month, custom date).
• Choose Payment Mode (Cash, Online, UPI, Bank Transfer).
• Select Class & Section (optional).
• Click Generate Report.
• Displays Receipt Number, Date, Student Name, Payment Mode, and Amount Paid.
• Click on a Receipt Number to view the detailed receipt.

• Click Download PDF to generate a printable receipt.

Due Fees Report
• Navigate to Reports > Due Fees Report
• Go to Reports > Due Fees Report.
• Select Class & Section or search for a student.
  User Manual
72

• Choose a Due Date (e.g., overdue for more than 30 days).
• Click Generate Report.
• Displays Student Name, Pending Amount, and Due Date.
• Click Send SMS/Email Reminder to notify parents about pending fees.

• Export the list using CSV/PDF for follow-up actions.
Fine Report
• Navigate to Reports > Fine Report
• Go to Reports > Fine Report.
• Select Class & Section.
• Choose a Fine Type (Late Fee Payment, Library Fine, Exam Fine).
• Click Generate Report.
• Shows Student Name, Fine Amount, Reason, and Payment Status.
• Click Mark as Paid if fine has been settled.
  User Manual
73


• Send fine reminders via SMS/Email.

Financial Reports
• Go to Reports > Financial Reports.
• Analyze financial data.

Account Statement
• Navigate to Reports > Financial Reports > Account Statement
• From the dashboard, go to Reports > Financial Reports.
• Click on Account Statement.
• Generate an Account Summary
• Select Account Type (School, Hostel, Transport, Other).
• Choose Date Range (custom or predefined periods like monthly, quarterly, yearly).
• Click Filter to view all transactions.
• Displays Opening Balance, Total Income, Total Expenses, and Closing Balance.
• Click Export (CSV/Excel/PDF) to save records.
  User Manual
74


Income Reports
• Navigate to Reports > Financial Reports > Income Reports
• Go to Reports > Financial Reports > Income Reports.
• Select Income Source (Fees, Donations, Grants, Other).
• Choose Date Range for the report.
• Click Filter.
• Displays Source, Date, Amount, and Payment Mode.
• Provides total income for the selected period.
• Export the report in CSV/PDF.
  User Manual
75


Expense Reports
• Navigate to Reports > Financial Reports > Expense Reports
• Go to Reports > Financial Reports > Expense Reports.
• Select Expense Category (Salaries, Infrastructure, Utility Bills, Maintenance).
• Choose Date Range.
• Click Generate Report.
• Expense Type, Amount, Date, and Payment Mode.
• Shows total expenditure for the selected period.
• Export for record-keeping and financial analysis.


  User Manual
76

Transactions Reports
• Navigate to Reports > Financial Reports > Transactions Reports
• Go to Reports > Financial Reports > Transactions Reports.
• Select Account Type (School, Hostel, Transport, Miscellaneous).
• Choose Date Range.
• Click Generate Report.
• Displays Transaction ID, Date, Type (Income/Expense), Amount, and Payment Mode.

• Helps track all money movements within the institution.
• Export the report in Excel/PDF.
Balance Sheet
• Navigate to Reports > Financial Reports > Balance Sheet
• Go to Reports > Financial Reports > Balance Sheet.
• Select Date Range (e.g., yearly financial summary).
• Click Generate Report.
• Displays Total Assets, Liabilities, and Net Balance.
  User Manual
77


• Helps evaluate the financial stability of the institution.
• Export the balance sheet for external audits.
Income vs Expense Report
• Navigate to Reports > Financial Reports > Income Vs Expense
• Go to Reports > Financial Reports > Income Vs Expense.
• Select Date Range (monthly, quarterly, yearly).
• Click Generate Report.
• Displays Total Income, Total Expenses, and Net Profit/Loss.

• Provides Graphical Representations (bar charts, pie charts).
  User Manual
78

• Helps in budgeting and financial planning.
Attendance Reports
• Navigate to Reports > Attendance Reports.
• View attendance statistics.
Student Attendance Reports
• Navigate to Reports > Attendance Reports > Student Reports
• Log in to VedCool as an administrator, branch admin, or teacher.
• From the dashboard, go to Reports > Attendance Reports.
• Click on Student Reports.
• Select Class & Section.
• Choose Date Range (daily, weekly, monthly, or custom).
• Select Attendance Type (Present, Absent, Late, Leave).
• Click Generate Report.
• Displays Student Name, Roll Number, Total Days, Present, Absent, and Percentage.

• Shows attendance trends in graphical format (optional).
• Export in Excel/PDF/CSV for record-keeping.
Employee Attendance Reports
• Navigate to Reports > Attendance Reports > Employee Reports
• Go to Reports > Attendance Reports > Employee Reports.
• Select Department & Designation (e.g., Teaching Staff, Admin, Support Staff).
• Choose Date Range (monthly, quarterly, yearly).
• Select Attendance Type (Present, Absent, Leave, Work-from-Home).
• Click Generate Report.
• Displays Employee Name, ID, Designation, Total Working Days, Absent Days, Leave Taken.
  User Manual
79


• Shows trends to monitor staff punctuality and leaves.
• Export for payroll and HR processing.
Exam Attendance Reports
• Navigate to Reports > Attendance Reports > Exam Reports
• Go to Reports > Attendance Reports > Exam Reports.
• Choose Exam Name (e.g., Mid-Term, Final Exam).
• elect Class & Section.
• Choose Date Range (for multi-day exams).
• Click Generate Report.
• Displays Student Name, Roll Number, Present/Absent Status, and Exam Date.

• Identifies students who missed exams for re-examination planning.
• Export the report in Excel/PDF for academic records.
Human Resource Report
• Go to Reports > HR Reports.
• Access HR-related reports.
• From the dashboard, go to HR > Payroll.
  User Manual
80

• Click on Payroll Summary.
• Choose Employee Type (Teaching Staff, Non-Teaching Staff, Support Staff).
• Select Month & Year for the payroll period.
• Choose Department (if needed).
• Click Filter

• Displays Employee Name, Employee ID, Basic Salary, Deductions, Bonuses, Net Pay.
• Shows Payroll Status (Pending, Processed, Paid).
• Allows export in Excel/PDF/CSV for accounting records.
Leave Reports
• Navigate to HR > Leave Management > Leave Reports
• Go to HR > Leave Management.
• Click on Leave Reports.
• Select Employee Type & Department.
• Choose Leave Type (Casual, Sick, Maternity, Unpaid, Work-from-Home).
• Set the Date Range (monthly, quarterly, yearly).
• Click Filter
• Displays Employee Name, ID, Total Leaves Taken, Remaining Leaves, Leave Status.
  User Manual
81


• Highlights employees with frequent leave patterns.
• Allows export for HR processing and compliance tracking.

Examination Reports
• Navigate to Reports > Examination Reports.
• View exam results and analytics.
Report Card Generation
• Log in to VedCool as an administrator, teacher, or examination officer.
• From the dashboard, go to Examination > Reports.





  User Manual
82


• Click on Report Card.
• Choose Exam Name (e.g., Mid-Term, Final Exam).
• Select Class & Section.
• Choose Student Name/Roll Number (or generate for all students).
• Click filter
• Displays Student Name, Roll Number, Subject-wise Marks, Grades, Remarks, and Overall
Percentage.
• Provides a Pass/Fail Status and teacher comments.
• Allows customization (e.g., add school logo, grading scale).
• Print or Export report cards in PDF/Excel.
Tabulation Sheet
• Navigate to Examination > Reports > Tabulation Sheet
• Go to Examination > Reports.
• Click on Tabulation Sheet.
• Choose Exam Name.
• Select Class & Section.
• Choose Subject-wise or Full Exam Report.
• Click Generate Report.
• Displays subject-wise marks of all students in a tabular format.
• Provides total marks, percentage, rank, and overall performance comparison.
• Highlights top-performing and underperforming students.
• Allows Excel/PDF export for record-keeping.
  User Manual
83



SETTINGS
The Settings module allows administrators to configure the system, including global settings, school-specific
details, user roles, session management, and system updates. It ensures the application works according to the
institution’s requirements.
Global Settings
• Navigate to Settings > Global.
• Manage global system settings.
• Set Timezone & Date Format
  User Manual
84

• Enable/Disable Features (e.g., Attendance, Exam Reports).
• Email & SMS Notifications (Enable alerts for fees, attendance, exams).

• Click Save Changes.
School Settings
• Go to Settings > School Settings.
• Customize school-specific settings.
• Enter School Name, Address, Contact Details.
• Upload School Logo & Header (for report cards, invoices).
• Set Academic Structure (standards, sections).
  User Manual
85


• Click Save Changes.
Role Permission
• Used to manage user roles and access levels.
• Navigate to Settings > Role Permission

• Go to Settings > Role Permission.
 Assign Permissions
• Select User Role (Admin, Teacher, Student, Parent, Accountant).
• Enable/Disable Module Access (Exams, Fees, Reports, Library).
  User Manual
86


• Click Save Permissions.
Session Settings
• Used to configure academic sessions.
• Navigate to Settings > Session Settings
• Go to Settings > Session Settings.
• Manage Sessions
• Create New Session (e.g., 2024-25).
• Set Start & End Dates.
• Mark Active Session for the current academic year.





  User Manual
87

• Click Save Session.
Translations
• Used to manage multi-language support.
• Navigate to Settings > Translations
• Go to Settings > Translations.
• Select Language & Customize
• Choose Default Language (English, Hindi, Spanish, etc.).
• Edit Custom Translations for module labels.

• Click Save Changes.
  User Manual
88

Cron Job
Used for automating tasks (e.g., sending reminders, backups).
• Navigate to Settings > Cron Job
• Go to Settings > Cron Job.
• Configure Automated Tasks
• Enable Auto Attendance Processing.
• Set Fee Payment Reminders frequency.
• Schedule Daily Database Backups.
• Click Activate Cron Job.

Custom Fields
Used for adding extra fields to forms (students, employees, fees).
• Navigate to Settings > Custom Fields
• Go to Settings > Custom Fields.


Add New Fields
• Choose Module (Students, Admissions, Employees, Fees).
• Add Field Name, Type (Text, Dropdown, Checkbox).

• Click Save Field.
Database Backup
Used for securing system data.
• Navigate to Settings > Database Backup
• Go to Settings > Database Backup.

Perform Backup
• Click Backup Now.
• Download the backup file (.sql format).
• Enable Auto Backup Schedule
  User Manual
90


System Update
Used to update VedCool to the latest version.
• Navigate to Settings > System Update
• Go to Settings > System Update.
• Check & Install Updates
• Click Check for Updates.
• If available, click Update Now.
• Ensure Database Backup is completed before updating.

Module Permission
The Module Permissions feature controls which users can access specific modules in VedCool.
Administrators can grant or restrict access based on user roles, ensuring appropriate access to sensitive data
and functionalities.
Adjust additional configurations. Accessing Module Permissions
• Navigate to Settings > Role Permission > Module Permission
• Log in as an administrator.
• Go to Settings > Role Permission.
• Click on Module Permission.
"""


# --- Helper Functions ---
def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    if len(text) > max_chars:
        logging.warning(f"Truncating text from {len(text)} to {max_chars} characters.")
        return text[:max_chars]
    return text

# --- Core Gemini API Functions with Tenacity Retries ---
@retry(
    wait=wait_random_exponential(min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((Exception,))
)
def get_embedding_with_retry(text: str, model: str = EMBEDDING_MODEL):
    if not text or not text.strip():
        logging.warning("Attempted to get embedding for empty text.")
        return None
    try:
        result = genai.embed_content(
            model=model,
            content=text,
            task_type="retrieval_document"
        )
        embedding = np.array(result['embedding'])
        if embedding.size == 0:
            logging.error("Received empty embedding from API.")
            return None
        return embedding
    except Exception as e:
        logging.error(f"Failed to generate embedding: {str(e)}")
        raise

# --- Manual Parsing Function ---
def parse_manual(manual_text_content: str):
    lines = manual_text_content.splitlines()
    try:
        toc_start_idx = next(i for i, line in enumerate(lines) if line.strip().upper() == "TABLE OF CONTENT")
    except StopIteration:
        logging.error("Table of Contents not found in manual.")
        return []

    toc_entry_pattern = re.compile(r"^(.*?)\s*\.{3,}\s*(\d+)\s*$")
    toc_lines_texts = []
    current_idx = toc_start_idx + 1
    max_toc_scan_lines = current_idx + 700
    consecutive_non_match_limit = 5
    non_match_count = 0
    meaningful_toc_entries_count = 0

    while current_idx < len(lines) and current_idx < max_toc_scan_lines:
        line_content = lines[current_idx].strip()
        if not line_content:
            current_idx += 1
            non_match_count = 0
            continue

        match = toc_entry_pattern.match(line_content)
        if match:
            heading_text_candidate = match.group(1).strip()
            if not heading_text_candidate.isdigit() and "user manual" not in heading_text_candidate.lower():
                toc_lines_texts.append(line_content)
                meaningful_toc_entries_count += 1
                non_match_count = 0
            else:
                non_match_count = 0
        else:
            if meaningful_toc_entries_count > 5:
                non_match_count += 1
                if non_match_count >= consecutive_non_match_limit:
                    logging.info(f"Stopping TOC scan at line {current_idx} after {consecutive_non_match_limit} non-matching lines.")
                    break
            elif not line_content.isupper() and len(line_content) > 60:
                non_match_count += 1
                if non_match_count >= 2 and meaningful_toc_entries_count < 3:
                    logging.info(f"Stopping TOC scan early due to non-matching lines with few entries found.")
                    break
        current_idx += 1

    if not toc_lines_texts:
        logging.error("No valid Table of Contents entries extracted.")
        return []

    extracted_headings_info = []
    for toc_line in toc_lines_texts:
        match = toc_entry_pattern.match(toc_line)
        if match:
            heading_text_candidate = match.group(1).strip()
            if "................................................................-xl" in heading_text_candidate:
                heading_text_candidate = heading_text_candidate.split("................................................................-xl")[0].strip()
            heading_text_candidate = re.sub(r'\s+User Manual\s*\d*$', '', heading_text_candidate, flags=re.IGNORECASE).strip()
            heading_text_candidate = re.sub(r'\s+\d+$', '', heading_text_candidate).strip()

            if heading_text_candidate and not heading_text_candidate.isdigit() and len(heading_text_candidate) > 2:
                extracted_headings_info.append(
                    (heading_text_candidate, heading_text_candidate.upper())
                )

    section_positions = []
    content_lines_stripped = [line.strip() for line in lines]
    content_lines_upper_stripped = [line.upper() for line in content_lines_stripped]
    content_search_start_offset = toc_start_idx + len(toc_lines_texts) + 1

    if extracted_headings_info:
        first_toc_heading_upper = extracted_headings_info[0][1]
        try:
            first_heading_actual_pos = -1
            for i in range(content_search_start_offset, len(content_lines_upper_stripped)):
                if content_lines_upper_stripped[i] == first_toc_heading_upper and len(content_lines_stripped[i]) < 150:
                    first_heading_actual_pos = i
                    break
            if first_heading_actual_pos != -1:
                content_search_start_offset = first_heading_actual_pos
                logging.info(f"Adjusted content search start offset based on first TOC heading.")
            else:
                logging.warning(f"First TOC heading not found after TOC.")
        except Exception as e:
            logging.error(f"Error finding first TOC heading position: {e}")

    current_search_line = content_search_start_offset
    found_headings_indices = set()

    for display_heading, match_heading_upper in extracted_headings_info:
        found_line_idx = -1
        try:
            for i in range(current_search_line, len(content_lines_upper_stripped)):
                if i in found_headings_indices:
                    continue
                if content_lines_upper_stripped[i] == match_heading_upper and len(content_lines_stripped[i]) < 150:
                    found_line_idx = i
                    found_headings_indices.add(i)
                    break
            if found_line_idx != -1:
                section_positions.append((display_heading, found_line_idx))
            else:
                logging.warning(f"Heading '{display_heading}' not found in manual content.")
        except Exception as e:
            logging.error(f"Error finding position for heading '{display_heading}': {e}")
        if found_line_idx != -1:
            current_search_line = found_line_idx + 1

    section_positions.sort(key=lambda x: x[1])

    parsed_sections = []
    for i in range(len(section_positions)):
        heading_display, start_line_idx_content = section_positions[i]
        content_block_start_line = start_line_idx_content + 1
        if i < len(section_positions) - 1:
            content_block_end_line = section_positions[i + 1][1]
        else:
            content_block_end_line = len(lines)

        actual_content_lines = lines[content_block_start_line:content_block_end_line]
        content_text = '\n'.join(ln.strip() for ln in actual_content_lines if ln.strip()).strip()
        content_text_lines = content_text.split('\n')
        cleaned_content_text_lines = [line for line in content_text_lines if not re.match(r"^\s*User Manual\s*\d*\s*$", line, flags=re.IGNORECASE)]
        content_text = '\n'.join(cleaned_content_text_lines).strip()

        if content_text:
            parsed_sections.append((heading_display, content_text))
        else:
            logging.info(f"Section '{heading_display}' resulted in no content after parsing.")

    logging.info(f"Successfully parsed {len(parsed_sections)} sections.")
    return parsed_sections

# --- Q&A Function ---
def answer_question(question: str, section_data: list, threshold=0.40, top_n=3):
    logging.info(f"Embedding question: '{question}'")
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=question,
            task_type="retrieval_query"
        )
        question_embedding = np.array(result['embedding'])
    except Exception as e:
        logging.error(f"Error generating question embedding: {str(e)}")
        return "I encountered an issue processing your question with the embedding model. Please try again."

    similarities = []
    for heading, content, embedding in section_data:
        if embedding is None or not isinstance(embedding, np.ndarray) or embedding.ndim == 0 or embedding.size == 0:
            logging.warning(f"Skipping section '{heading}' due to invalid or empty embedding.")
            continue
        try:
            similarity = 1 - cosine(question_embedding, embedding)
            similarities.append((similarity, heading, content))
        except Exception as e:
            logging.error(f"Error calculating cosine similarity for section '{heading}': {str(e)}")
            continue

    if not similarities:
        logging.warning("No sections with valid embeddings available.")
        return "The user manual content could not be searched at this time due to an issue with section embeddings."

    similarities.sort(key=lambda x: x[0], reverse=True)
    logging.info(f"Top {top_n} similarities for question '{question}':")
    for i, (sim_score, head, _) in enumerate(similarities[:top_n]):
        logging.info(f"  {i+1}. Similarity: {sim_score:.4f} with Section: '{head}'")

    relevant_sections_info = []
    for sim_score, heading, content in similarities[:top_n]:
        if sim_score >= threshold:
            relevant_sections_info.append({
                "heading": heading,
                "content": content,
                "similarity": sim_score
            })
        else:
            break

    if not relevant_sections_info:
        highest_sim_score = similarities[0][0] if similarities else -1.0
        logging.info(f"No sections found above threshold {threshold}. Highest similarity: {highest_sim_score:.4f}.")
        return "I've searched the VedCool user manual, but I couldn't find specific information that directly addresses your question in the available excerpts."

    combined_context = ""
    log_message_context_parts = []
    for i, section_info in enumerate(relevant_sections_info):
        combined_context += (
            f"MANUAL SECTION {i+1} TITLE: \"{section_info['heading']}\" (Similarity: {section_info['similarity']:.4f})\n"
            f"SECTION {i+1} CONTENT:\n\"\"\"\n{section_info['content']}\n\"\"\"\n\n"
        )
        log_message_context_parts.append(f"'{section_info['heading']}' (Sim: {section_info['similarity']:.4f})")

    prompt_for_llm = (
        f"You are a professional and helpful AI assistant for the VedCool platform. Your goal is to provide clear, concise, and easy-to-understand answers to user questions based *exclusively* on the provided excerpts from the VedCool user manual.\n\n"
        f"Follow these instructions carefully:\n"
        f"1. Base your answer *only* on the text provided in the 'CONTEXT FROM MANUAL' section(s) below.\n"
        f"2. Answer the 'USER'S QUESTION' concisely and accurately.\n"
        f"3. If the answer is found across multiple provided sections, synthesize the information smoothly.\n"
        f"4. If the provided context directly answers the question, provide the answer directly.\n"
        f"5. If the provided context mentions the topic but does not contain the specific details to fully answer the question, state what information is available and what is missing.\n"
        f"6. If the provided context does not contain any relevant information to answer the question, clearly state that the information is not found in the provided excerpts of the manual.\n"
        f"7. Do not use any outside knowledge or make assumptions beyond the provided text.\n"
        f"8. Present answers in a clear, well-formatted way. Use bullet points for steps or lists if appropriate.\n\n"
        f"CONTEXT FROM MANUAL:\n{combined_context}\n\n"
        f"USER'S QUESTION: \"{question}\"\n\n"
        f"PROFESSIONAL AND CLEAR ANSWER:"
    )
    logging.info(f"Generating response using section(s): {', '.join(log_message_context_parts)}")
    response = generate_response_with_retry(prompt=prompt_for_llm)
    return response

# --- Main Execution ---
if __name__ == "__main__":
    parsed_manual_sections = parse_manual(manual_text)
    if not parsed_manual_sections:
        logging.error("No sections parsed from manual. Exiting.")
        sys.exit(1)

    section_data_for_chatbot = []
    if os.path.exists(EMBEDDINGS_CACHE_FILE):
        try:
            with open(EMBEDDINGS_CACHE_FILE, 'rb') as f:
                loaded_data = pickle.load(f)
            if isinstance(loaded_data, list) and all(isinstance(item, tuple) and len(item) == 3 for item in loaded_data):
                section_data_for_chatbot = loaded_data
                logging.info(f"Loaded {len(section_data_for_chatbot)} embeddings from cache.")
            else:
                logging.warning("Cached data invalid. Recomputing embeddings.")
        except Exception as e:
            logging.error(f"Error loading embeddings cache: {str(e)}")
            section_data_for_chatbot = []

    if not section_data_for_chatbot:
        logging.info("Computing embeddings for manual sections...")
        for i, (heading, content) in enumerate(parsed_manual_sections):
            logging.info(f"Processing section {i+1}/{len(parsed_manual_sections)}: '{heading}'")
            text_to_embed = f"Section Title: {heading}\n\nContent:\n{content}"
            truncated_text = truncate_text_to_tokens(text_to_embed, MAX_TOKENS_FOR_EMBEDDING)
            if len(truncated_text) < len(text_to_embed):
                logging.warning(f"Truncated section '{heading}' from {len(text_to_embed)} to {len(truncated_text)} chars.")
            if not truncated_text.strip():
                logging.warning(f"Skipping empty section '{heading}' after truncation.")
                continue
            try:
                embedding_array = get_embedding_with_retry(text=truncated_text)
                if embedding_array is not None and embedding_array.size > 0:
                    section_data_for_chatbot.append((heading, content, embedding_array))
                else:
                    logging.warning(f"Failed to compute embedding for section: {heading}.")
            except Exception as e:
                logging.error(f"Failed to embed section '{heading}': {str(e)}")
                continue

        if section_data_for_chatbot:
            try:
                with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
                    pickle.dump(section_data_for_chatbot, f)
                logging.info(f"Embeddings saved to {EMBEDDINGS_CACHE_FILE}")
            except Exception as e:
                logging.error(f"Error saving embeddings: {str(e)}")
        else:
            logging.error("No embeddings computed. Chatbot may not function correctly.")

    if section_data_for_chatbot:
        print("\nVedCool Chatbot ready! Ask your question.")
        while True:
            try:
                question = input("Your question: ").strip()
                if question.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                if not question:
                    print("Please enter a valid question.")
                    continue
                answer = answer_question(question, section_data_for_chatbot, threshold=0.40, top_n=3)
                print(f"\nResponse:\n{answer}\n")
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logging.error(f"Unexpected error: {str(e)}")
                print("An error occurred. Please try again.")
    else:
        logging.error("No section data available. Exiting.")
        print("Error: No section data available. Check logs for errors.")
