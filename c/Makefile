CC      ?= gcc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -Werror
BUILD   := build

$(BUILD)/test_policy: test_policy.c rl_controller.c rl_controller.h policy_weights.h
	mkdir -p $(BUILD)
	$(CC) $(CFLAGS) -o $@ test_policy.c rl_controller.c -lm

test: $(BUILD)/test_policy
	$(BUILD)/test_policy vectors test_vectors.csv

clean:
	rm -rf $(BUILD)

.PHONY: test clean
