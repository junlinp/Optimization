#include "gtest/gtest.h"
#include "JET.h"

TEST(JET, Plus) {
    Jet<2> a(1.0, 0);
    Jet<2> b(2.0, 1);

    Jet<2> c = a + b;
    auto gradient = c.Gradient();
    EXPECT_EQ(gradient(0), 1.0);
    EXPECT_EQ(gradient(1), 1.0);
}
int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
