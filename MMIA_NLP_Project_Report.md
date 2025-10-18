# Proyecto MMIA-NLP
## Implementación y Evaluación de Métodos de Adaptación de Modelos de Lenguaje

### Resumen

Este informe documenta el trabajo realizado en el proyecto MMIA-NLP, que consistió en la implementación completa de un sistema de análisis de sentimientos basado en el modelo Llama, seguido de la evaluación de tres estrategias de adaptación: Zero-shot Prompting, Fine-tuning Completo y LoRA Fine-tuning. El proyecto se desarrolló sobre dos datasets especializados: SST (Stanford Sentiment Treebank) y CFIMDB (Counterfactual IMDB), proporcionando evidencia empírica sobre la efectividad relativa de cada método en tareas de clasificación de sentimientos.

## 1. Trabajo de Implementación Realizado

### 1.1 Arquitectura del Sistema Implementado

El proyecto requirió la implementación de un sistema de análisis de sentimientos basado en el modelo Llama. La arquitectura implementada incluye:

#### Componentes principales:
- **Modelo Base Llama**: Arquitectura Transformer completa con múltiples capas
- **Clasificador de Embeddings**: LlamaEmbeddingClassifier para tareas de clasificación
- **Optimizador AdamW**: Implementación completa con gradient clipping y weight decay
- **Embeddings Posicionales RoPE**: Rotary Position Embeddings para mejor comprensión de secuencias
- **Adaptación Eficiente LoRA**: Low-Rank Adaptation para fine-tuning eficiente

#### Funcionalidades implementadas:
- **Generación de texto** con diferentes estrategias de sampling (greedy, temperature, epsilon)
- **Clasificación de sentimientos** con fine-tuning completo y LoRA
- **Zero-shot prompting** para evaluación sin entrenamiento previo
- **Sistema de entrenamiento** con monitoreo de métricas y early stopping

### 1.2 Modificaciones realizadas

El trabajo de implementación involucró modificaciones en 7 archivos del repositorio, transformando un esqueleto de código con `NotImplementedError`:

#### **`classifier.py` - Clasificador de Embeddings**
Implementación del método `forward()` para clasificación de sentimientos
```python
# Implementación del pipeline completo:
# Llama → hidden states → dropout → classifier → log-softmax
_, hidden_states = self.llama(input_ids)
final_hidden_state = hidden_states[:, -1, :]
dropped_hidden = self.dropout(final_hidden_state)
logits = self.classifier_head(dropped_hidden)
log_probs = F.log_softmax(logits, dim=-1)
```

#### **`llama.py` - Modelo Base Llama**
Implementación de la arquitectura Transformer
- **LayerNorm**: Normalización por capas con fórmula matemática completa
- **Attention**: Mecanismo de atención multi-cabeza con softmax y dropout
- **LlamaLayer**: Capa completa con conexiones residuales y feed-forward
- **Sampling**: Estrategias de generación (greedy, temperature, epsilon sampling)

#### **`lora.py` - Low-Rank Adaptation**
Implementación de adaptación eficiente con matrices de bajo rango
```python
# Implementación del path LoRA: x → A → B → scale
lora_intermediate = torch.matmul(x, self.lora_A.t())
lora_output = torch.matmul(lora_intermediate, self.lora_B.t())
lora_output = lora_output * self.scaling
final_output = original_output + lora_output
```

#### **`optimizer.py` - Optimizador AdamW**
Implementación del optimizador con todas sus características
- **Gradient Clipping**: Prevención de gradient explosion
- **Bias Correction**: Corrección de sesgo para momentos
- **Weight Decay**: Regularización L2 integrada
- **Estado del optimizador**: Mantenimiento de momentos y contadores

#### **`rope.py` - Rotary Position Embeddings**
Implementación de embeddings posicionales rotatorios
- **Cálculo de frecuencias**: θ^(-2i/d) para cada dimensión
- **Matrices de rotación**: Aplicación de rotaciones complejas
- **Integración con atención**: Mejora de comprensión posicional

#### **`run_llama.py` - Sistema de Ejecución**
Mejoras en compatibilidad
- **Encoding UTF-8**: Manejo correcto de caracteres Unicode
- **Carga de modelos**: Parámetros seguros para torch.load()
- **Compatibilidad**: Funcionamiento en diferentes sistemas operativos

#### **`sanity_check.py` - Verificación y Debugging**
Sistema de verificación y debugging mejorado
- **Modo evaluación**: Asegurar consistencia en inferencia
- **Comparaciones detalladas**: Verificación de implementación
- **Debugging informativo**: Análisis de diferencias entre implementaciones

## 2. Trabajo realizado

### 2.1 Diseño Experimental

El trabajo se diseñó para evaluar tres estrategias de adaptación de modelos de lenguaje en tareas de clasificación de sentimientos. Se implementó un pipeline experimental que incluye:

#### **Estrategias Evaluadas:**
1. **Zero-shot Prompting**: Evaluación sin entrenamiento previo
2. **Fine-tuning Completo**: Adaptación de todos los parámetros del modelo
3. **LoRA Fine-tuning**: Adaptación eficiente con matrices de bajo rango

#### **Datasets Utilizados:**
- **SST (Stanford Sentiment Treebank)**: Dataset estándar de análisis de sentimientos
- **CFIMDB (Counterfactual IMDB)**: Dataset más complejo con ejemplos contrafactuales

#### **Métricas de Evaluación:**
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase
- **Recall**: Sensibilidad por clase
- **F1-Score**: Media armónica de precisión y recall

### 2.2 Resultados Experimentales Obtenidos

#### **Tabla de Resultados Principales**

| Dataset | Método | Dev Accuracy | Test Accuracy | Mejora vs Zero-shot |
|---------|--------|--------------|---------------|-------------------|
| **SST** | Zero-shot | 0.244 | 0.248 | - |
| **SST** | Fine-tuning | 0.423 | 0.417 | +68.1% |
| **SST** | LoRA | 0.273 | 0.263 | +6.0% |
| **CFIMDB** | Zero-shot | 0.482 | 0.131 | - |
| **CFIMDB** | Fine-tuning | 0.837 | 0.471 | +259.5% |
| **CFIMDB** | LoRA | 0.506 | 0.318 | +142.7% |

#### **Métricas Detalladas por Experimento**

| Dataset | Método | Accuracy | Precision | Recall | F1-Score |
|---------|--------|----------|-----------|--------|----------|
| **SST** | Zero-shot | 0.244 | 0.242 | 0.244 | 0.212 |
| **SST** | Fine-tuning | 0.423 | 0.441 | 0.423 | 0.389 |
| **SST** | LoRA | 0.273 | 0.218 | 0.273 | 0.207 |
| **CFIMDB** | Zero-shot | 0.482 | 0.469 | 0.482 | 0.415 |
| **CFIMDB** | Fine-tuning | 0.837 | 0.837 | 0.837 | 0.837 |
| **CFIMDB** | LoRA | 0.506 | 0.508 | 0.506 | 0.492 |

### 2.3 Análisis de Consistencia y Overfitting

#### **Gap Dev-Test por Método:**
- **SST Zero-shot**: 0.004 (Estable)
- **SST Fine-tuning**: 0.006 (Estable)
- **SST LoRA**: 0.010 (Estable)
- **CFIMDB Zero-shot**: 0.351 (Inconsistente)
- **CFIMDB Fine-tuning**: 0.366 (Overfitting detectado)
- **CFIMDB LoRA**: 0.188 (Moderadamente estable)

#### **Observaciones Importantes:**
1. **CFIMDB muestra mayor variabilidad** entre desarrollo y test
2. **Fine-tuning en CFIMDB presenta overfitting** significativo
3. **SST mantiene consistencia** en todos los métodos
4. **LoRA muestra mejor generalización** que fine-tuning completo en CFIMDB

## 3. Análisis de Resultados y Conclusiones

### 3.1 Análisis Comparativo por Método

#### **Fine-tuning Completo - Mejor Rendimiento Absoluto**
**Resultados obtenidos:**
- **SST**: 0.417 accuracy (test) - Mejora del 68.1% sobre zero-shot
- **CFIMDB**: 0.471 accuracy (test) - Mejora del 259.5% sobre zero-shot
- **Consistencia**: Gap dev-test mínimo en SST (0.006), pero overfitting en CFIMDB (0.366)

**Interpretación:**
- El fine-tuning completo aprovecha toda la capacidad del modelo
- En SST, muestra generalización adecuada con gap mínimo
- En CFIMDB, presenta overfitting significativo, sugiriendo necesidad de regularización

#### **LoRA Fine-tuning - Balance Eficiencia-Rendimiento**
**Resultados obtenidos:**
- **SST**: 0.263 accuracy (test) - Mejora del 6.0% sobre zero-shot
- **CFIMDB**: 0.318 accuracy (test) - Mejora del 142.7% sobre zero-shot
- **Generalización**: Mejor gap dev-test en CFIMDB (0.188) que fine-tuning completo

**Interpretación:**
- LoRA proporciona adaptación eficiente con mejor generalización
- Menor overfitting que fine-tuning completo en datasets complejos
- Balance óptimo entre rendimiento y eficiencia computacional

#### **Zero-shot Prompting - Línea Base**
**Resultados obtenidos:**
- **SST**: 0.248 accuracy (test) - Rendimiento base consistente
- **CFIMDB**: 0.131 accuracy (test) - Rendimiento muy bajo
- **Variabilidad**: Gap significativo en CFIMDB (0.351)

**Interpretación:**
- Zero-shot funciona mejor en SST que en CFIMDB
- CFIMDB requiere adaptación específica del modelo

### 3.2 Análisis por Dataset

#### **SST (Stanford Sentiment Treebank) - Comportamiento Estable**
- **Consistencia**: Todos los métodos muestran gap dev-test < 0.01
- **Rendimiento moderado**: Máximo 0.417 accuracy con fine-tuning
- **Diferencias graduales**: Progresión suave entre métodos

**Implicaciones:**
- SST es un dataset más "predecible" para el modelo
- Menor complejidad inherente permite mejor generalización

#### **CFIMDB (Counterfactual IMDB) - Comportamiento Variable**
- **Mayor variabilidad**: Gap dev-test significativo en todos los métodos
- **Mejor beneficio del fine-tuning**: Mejora del 259.5% vs zero-shot
- **Overfitting**: Fine-tuning muestra gap de 0.366

**Implicaciones:**
- CFIMDB es más complejo y requiere adaptación específica
- Fine-tuning muestra mayor ventaja relativa
- Necesidad de técnicas de regularización para evitar overfitting

### 3.3 Detección y Análisis de Overfitting

#### **Caso Crítico: CFIMDB Fine-tuning**
**Evidencia de overfitting:**
- **Gap dev-test**: 0.366 (0.837 dev vs 0.471 test)
- **Causa probable**: Sobreajuste al conjunto de desarrollo

#### **Casos Estables: SST y LoRA**
**Evidencia de generalización:**
- **SST**: Todos los métodos muestran gap < 0.01
- **LoRA en CFIMDB**: Gap moderado (0.188) vs fine-tuning (0.366)

## 4. Conclusiones y Recomendaciones del Trabajo Realizado

### 4.1 Conclusiones Técnicas Principales

#### **1. Efectividad Relativa de Métodos - Evidencia Empírica**
Basado en los resultados experimentales obtenidos:

1. **Fine-tuning Completo** demuestra ser el método más efectivo en términos absolutos:
   - **Mejor rendimiento**: 0.471 accuracy en CFIMDB, 0.417 en SST
   - **Mayor mejora relativa**: +259.5% en CFIMDB, +68.1% en SST
   - **Limitación**: Overfitting significativo en CFIMDB (gap 0.366)

2. **LoRA Fine-tuning** ofrece el mejor balance generalización-rendimiento:
   - **Rendimiento intermedio**: 0.318 accuracy en CFIMDB, 0.263 en SST
   - **Mejor generalización**: Gap dev-test 0.188 en CFIMDB vs 0.366 de fine-tuning

3. **Zero-shot Prompting** muestra limitaciones significativas:
   - **Rendimiento base**: 0.131 accuracy en CFIMDB, 0.248 en SST
   - **Variabilidad**: Gap dev-test 0.351 en CFIMDB

#### **2. Diferencias entre Datasets**
**Análisis basado en los resultados obtenidos:**

- **SST (Stanford Sentiment Treebank)**:
  - **Comportamiento estable**: Gap dev-test < 0.01 en todos los métodos
  - **Rendimiento moderado**: Máximo 0.417 accuracy

- **CFIMDB (Counterfactual IMDB)**:
  - **Mayor variabilidad**: Gap dev-test significativo en todos los métodos
  - **Mejor beneficio del fine-tuning**: Mejora del 259.5% vs zero-shot

#### **3. Detección de Overfitting - Análisis Crítico**
**Evidencia empírica de los resultados:**

- **Caso crítico identificado**: CFIMDB Fine-tuning
  - **Gap dev-test**: 0.366 (0.837 dev vs 0.471 test)
  - **Causa**: Sobreajuste al conjunto de desarrollo
