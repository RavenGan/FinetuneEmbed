rm(list = ls())
set.seed(7)

library(dplyr)
library(tidyr)
library(ggplot2)

Fig1_A_lr <- data.frame(
  Model = c(
    "NoInstruct–small–Embedding–v0",
    "e5–small",
    "gte–tiny",
    "stella–base–en–v2",
    "e5–small–v2",
    "GIST–small–Embedding–v0",
    "OpenAI",
    "GIST–all–MiniLM–L6–v2",
    "bge–small–en–v1.5",
    "MedEmbed–small–v0.1",
    "gte–small"
  ),
  Task1 = c(
    0.728, 0.732, 0.706, 0.702, 0.692, 0.641, 0.712, 0.608, 0.629, 0.612, 0.629
  ),
  Task2 = c(
    0.908, 0.898, 0.904, 0.885, 0.879, 0.887, 0.863, 0.887, 0.886, 0.892, 0.866
  ),
  Task3 = c(
    0.935, 0.933, 0.930, 0.934, 0.915, 0.940, 0.931, 0.940, 0.924, 0.925, 0.914
  ),
  Task4 = c(
    0.907, 0.892, 0.912, 0.927, 0.922, 0.917, 0.875, 0.912, 0.905, 0.910, 0.915
  )
)

rownames(Fig1_A_lr) <- Fig1_A_lr$Model

Fig1_B_rf <- data.frame(
  Model = c(
    "stella–base–en–v2",
    "e5–small",
    "gte–tiny",
    "e5–small–v2",
    "bge–small–en–v1.5",
    "NoInstruct–small–Embedding–v0",
    "MedEmbed–small–v0.1",
    "OpenAI",
    "GIST–small–Embedding–v0",
    "GIST–all–MiniLM–L6–v2",
    "gte–small"
  ),
  Task1 = c(
    0.761, 0.730, 0.716, 0.745, 0.678, 0.673, 0.635, 0.664, 0.635, 0.632, 0.605
  ),
  Task2 = c(
    0.910, 0.908, 0.897, 0.908, 0.905, 0.897, 0.909, 0.889, 0.888, 0.889, 0.887
  ),
  Task3 = c(
    0.942, 0.918, 0.917, 0.903, 0.914, 0.930, 0.924, 0.931, 0.924, 0.928, 0.894
  ),
  Task4 = c(
    0.907, 0.889, 0.909, 0.872, 0.910, 0.893, 0.924, 0.874, 0.909, 0.856, 0.889
  )
)

rownames(Fig1_B_rf) <- Fig1_B_rf$Model

Fig2_A_lr <- data.frame(
  Model = c(
    "e5–small",
    "NoInstruct–small–Embedding–v0",
    "GIST–small–Embedding–v0",
    "stella–base–en–v2",
    "gte–tiny",
    "e5–small–v2",
    "MedEmbed–small–v0.1",
    "gte–small",
    "bge–small–en–v1.5",
    "GIST–all–MiniLM–L6–v2",
    "OpenAI"
  ),
  Task1 = c(
    0.772, 0.739, 0.731, 0.723, 0.723, 0.689, 0.709, 0.706, 0.700, 0.623, 0.674
  ),
  Task2 = c(
    0.928, 0.905, 0.897, 0.919, 0.904, 0.912, 0.901, 0.902, 0.900, 0.917, 0.896
  ),
  Task3 = c(
    0.921, 0.934, 0.938, 0.915, 0.919, 0.921, 0.919, 0.905, 0.911, 0.943, 0.930
  ),
  Task4 = c(
    0.883, 0.910, 0.902, 0.907, 0.905, 0.917, 0.892, 0.907, 0.885, 0.892, 0.870
  )
)

rownames(Fig2_A_lr) <- Fig2_A_lr$Model

Fig2_B_rf <- data.frame(
  Model = c(
    "stella–base–en–v2",
    "bge–small–en–v1.5",
    "e5–small–v2",
    "e5–small",
    "GIST–small–Embedding–v0",
    "MedEmbed–small–v0.1",
    "gte–tiny",
    "gte–small",
    "OpenAI",
    "NoInstruct–small–Embedding–v0",
    "GIST–all–MiniLM–L6–v2"
  ),
  Task1 = c(
    0.716, 0.717, 0.765, 0.718, 0.698, 0.639, 0.690, 0.661, 0.666, 0.599, 0.535
  ),
  Task2 = c(
    0.908, 0.911, 0.900, 0.921, 0.909, 0.912, 0.906, 0.898, 0.890, 0.910, 0.898
  ),
  Task3 = c(
    0.951, 0.925, 0.894, 0.904, 0.926, 0.930, 0.903, 0.907, 0.935, 0.913, 0.943
  ),
  Task4 = c(
    0.901, 0.909, 0.890, 0.894, 0.886, 0.914, 0.882, 0.909, 0.866, 0.895, 0.866
  )
)

rownames(Fig2_B_rf) <- Fig2_B_rf$Model

Fig3_finetune <- data.frame(
  Model = c(
    "stella–base–en–v2",
    "NoInstruct–small–Embedding–v0",
    "GIST–small–Embedding–v0",
    "e5–small–v2",
    "gte–small",
    "GIST–all–MiniLM–L6–v2",
    "bge–small–en–v1.5",
    "MedEmbed–small–v0.1",
    "gte–tiny",
    "e5–small"
  ),
  Task1 = c(
    0.680, 0.592, 0.588, 0.651, 0.600, 0.609, 0.583, 0.549, 0.575, 0.614
  ),
  Task2 = c(
    0.877, 0.888, 0.879, 0.867, 0.880, 0.886, 0.873, 0.883, 0.868, 0.868
  ),
  Task3 = c(
    0.869, 0.916, 0.924, 0.870, 0.901, 0.881, 0.915, 0.916, 0.899, 0.880
  ),
  Task4 = c(
    0.897, 0.912, 0.910, 0.905, 0.897, 0.895, 0.897, 0.905, 0.897, 0.860
  )
)

rownames(Fig3_finetune) <- Fig3_finetune$Model
models_used <- rownames(Fig3_finetune)

Fig1_A_lr <- Fig1_A_lr[models_used, ]
Fig1_B_rf <- Fig1_B_rf[models_used, ]
Fig2_A_lr <- Fig2_A_lr[models_used, ]
Fig2_B_rf <- Fig2_B_rf[models_used, ]

# remove the column named Model
Fig3_finetune <- Fig3_finetune[, -1]
Fig1_A_lr <- Fig1_A_lr[, -1]
Fig1_B_rf <- Fig1_B_rf[, -1]
Fig2_A_lr <- Fig2_A_lr[, -1]
Fig2_B_rf <- Fig2_B_rf[, -1]

Diff_3minus1_A_lr <- Fig3_finetune - Fig1_A_lr
Diff_3minus1_B_rf <- Fig3_finetune - Fig1_B_rf
Diff_3minus2_A_lr <- Fig3_finetune - Fig2_A_lr
Diff_3minus2_B_rf <- Fig3_finetune - Fig2_B_rf

# make plots
Diff_3minus1_A_lr$Model <- rownames(Diff_3minus1_A_lr)
# Convert to long format
df_long <- Diff_3minus1_A_lr %>%
  pivot_longer(cols = starts_with("Task"), names_to = "Task", values_to = "Delta")

# Plot heatmap
pdf("./res/2025_0625_Plots/Diff_3minus1_A_lr.pdf", width = 5.5, height = 6)
ggplot(df_long, aes(x = Task, y = Model, fill = Delta)) +
  geom_tile(color = "white", width = 0.9, height = 0.9) +
  geom_text(aes(label = sprintf("%.3f", Delta)), size = 4, color = "black") +
  scale_fill_gradient2(
    low = "#4682b4", mid = "white", high = "#d73027",
    midpoint = 0, name = NULL,
    guide = guide_colourbar(
      barwidth = 10,     # Length of the bar
      barheight = 0.5,   # Thickness of the bar
      title.position = "top"
    )
  ) +
  scale_x_discrete(position = "top") + 
  theme_minimal(base_size = 12) +
  labs(title = NULL, x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )
dev.off()

Diff_3minus1_B_rf$Model <- rownames(Diff_3minus1_B_rf)
# Convert to long format
df_long <- Diff_3minus1_B_rf %>%
  pivot_longer(cols = starts_with("Task"), names_to = "Task", values_to = "Delta")

# Plot heatmap
pdf("./res/2025_0625_Plots/Diff_3minus1_B_rf.pdf", width = 5.5, height = 6)
ggplot(df_long, aes(x = Task, y = Model, fill = Delta)) +
  geom_tile(color = "white", width = 0.9, height = 0.9) +
  geom_text(aes(label = sprintf("%.3f", Delta)), size = 4, color = "black") +
  scale_fill_gradient2(
    low = "#4682b4", mid = "white", high = "#d73027",
    midpoint = 0, name = NULL,
    guide = guide_colourbar(
      barwidth = 10,     # Length of the bar
      barheight = 0.5,   # Thickness of the bar
      title.position = "top"
    )
  ) +
  scale_x_discrete(position = "top") + 
  theme_minimal(base_size = 12) +
  labs(title = NULL, x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )
dev.off()


Diff_3minus2_A_lr$Model <- rownames(Diff_3minus2_A_lr)
# Convert to long format
df_long <- Diff_3minus2_A_lr %>%
  pivot_longer(cols = starts_with("Task"), names_to = "Task", values_to = "Delta")

# Plot heatmap
pdf("./res/2025_0625_Plots/Diff_3minus2_A_lr.pdf", width = 5.5, height = 6)
ggplot(df_long, aes(x = Task, y = Model, fill = Delta)) +
  geom_tile(color = "white", width = 0.9, height = 0.9) +
  geom_text(aes(label = sprintf("%.3f", Delta)), size = 4, color = "black") +
  scale_fill_gradient2(
    low = "#4682b4", mid = "white", high = "#d73027",
    midpoint = 0, name = NULL,
    guide = guide_colourbar(
      barwidth = 10,     # Length of the bar
      barheight = 0.5,   # Thickness of the bar
      title.position = "top"
    )
  ) +
  scale_x_discrete(position = "top") + 
  theme_minimal(base_size = 12) +
  labs(title = NULL, x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )
dev.off()


Diff_3minus2_B_rf$Model <- rownames(Diff_3minus2_B_rf)
# Convert to long format
df_long <- Diff_3minus2_B_rf %>%
  pivot_longer(cols = starts_with("Task"), names_to = "Task", values_to = "Delta")

# Plot heatmap
pdf("./res/2025_0625_Plots/Diff_3minus2_B_rf.pdf", width = 5.5, height = 6)
ggplot(df_long, aes(x = Task, y = Model, fill = Delta)) +
  geom_tile(color = "white", width = 0.9, height = 0.9) +
  geom_text(aes(label = sprintf("%.3f", Delta)), size = 4, color = "black") +
  scale_fill_gradient2(
    low = "#4682b4", mid = "white", high = "#d73027",
    midpoint = 0, name = NULL,
    guide = guide_colourbar(
      barwidth = 10,     # Length of the bar
      barheight = 0.5,   # Thickness of the bar
      title.position = "top"
    )
  ) +
  scale_x_discrete(position = "top") + 
  theme_minimal(base_size = 12) +
  labs(title = NULL, x = NULL, y = NULL) +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    panel.grid = element_blank(),
    legend.position = "bottom"
  )
dev.off()