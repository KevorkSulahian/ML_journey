#include <stdio.h>

int main() {
    char line[256];       // Buffer for input
    char opcode;          // Operation code
    float value;          // Operand value
    float display = 0.0;  // Accumulator initialized to 0

    while (fgets(line, 256, stdin) != NULL) {
        // Use sscanf to parse the input string
        sscanf(line, "%c %f", &opcode, &value);
        
        // Stop program when 'S' is encountered
        if (opcode == 'S') break;

        // Perform operation based on the opcode
        switch (opcode) {
            case '=':
                display = value;  // Set the accumulator
                break;
            case '+':
                display += value; // Add to the accumulator
                break;
            case '-':
                display -= value; // Subtract from the accumulator
                break;
            case '*':
                display *= value; // Multiply the accumulator
                break;
            case '/':
                if (value != 0) { // Check for division by zero
                    display /= value; // Divide the accumulator
                } else {
                    printf("Error: Division by zero\n");
                    continue;
                }
                break;
            default:
                printf("Error: Unknown command '%c'\n", opcode);
                continue;
        }

        // Print the current value of the accumulator
        printf("Display: %.2f\n", display);
    }

    return 0;
}
