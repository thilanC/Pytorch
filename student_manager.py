# Student Grade Manager

class Student:
    def __init__(self, name):
        self.name = name
        self.grades = []
    
    def add_grade(self, grade):
        self.grades.append(grade)
    
    def average(self):
        if len(self.grades) == 0:
            return 0
        return sum(self.grades) / len(self.grades)
    
    def __str__(self):
        return f"{self.name} | Grades: {self.grades} | Avg: {self.average():.2f}"


class GradeManager:
    def __init__(self):
        self.students = {}
    
    def add_student(self, name):
        if name not in self.students:
            self.students[name] = Student(name)
            print(f"✅ Student {name} added.")
        else:
            print("⚠️ Student already exists.")
    
    def add_grade(self, name, grade):
        if name in self.students:
            self.students[name].add_grade(grade)
            print(f"✅ Grade {grade} added for {name}.")
        else:
            print("⚠️ Student not found.")
    
    def show_students(self):
        if not self.students:
            print("⚠️ No students available.")
        for student in self.students.values():
            print(student)


def main():
    manager = GradeManager()
    
    while True:
        print("\n--- Student Grade Manager ---")
        print("1. Add Student")
        print("2. Add Grade")
        print("3. Show All Students")
        print("4. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == "1":
            name = input("Enter student name: ")
            manager.add_student(name)
        
        elif choice == "2":
            name = input("Enter student name: ")
            grade = float(input("Enter grade: "))
            manager.add_grade(name, grade)
        
        elif choice == "3":
            manager.show_students()
        
        elif choice == "4":
            print("Exiting program...")
            break
        
        else:
            print("⚠️ Invalid choice. Try again.")


if __name__ == "__main__":
    main()
