#include <gtest/gtest.h>
#include <cstdlib>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

TEST(ExternalConicBenchmark, ObjectivesAndKkt) {
  const char* srcdir = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  ASSERT_NE(srcdir, nullptr);
  ASSERT_NE(workspace, nullptr);
  const std::string script = std::string(srcdir) + "/" + workspace +
                             "/programing/external_conic_benchmark.py";
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    execlp("python3", "python3", script.c_str(), "--extended", "--verify", static_cast<char*>(nullptr));
    _exit(127);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
