import os
import sqlite3
import time
import datetime
import json
import math
import base64
import pandas as pd
import streamlit as st
import anthropic
import voyageai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- Configuración Inicial ---
load_dotenv()
cliente_claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
cliente_voyage = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

# Ruta Absoluta a la Base de Datos y Carpeta de Adjuntos
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'augmentiq_poc.db')
ADJUNTOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adjuntos')
if not os.path.exists(ADJUNTOS_DIR):
    os.makedirs(ADJUNTOS_DIR)

# --- MIGRACIÓN SILENCIOSA DE BD (Protege la data existente) ---
def asegurar_columnas_bitacora():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    try:
        cur.execute("ALTER TABLE tickets ADD COLUMN bitacora_tecnica TEXT DEFAULT ''")
        cur.execute("ALTER TABLE tickets ADD COLUMN ordenes_transporte TEXT DEFAULT ''")
        cur.execute("ALTER TABLE tickets ADD COLUMN objetos_sap TEXT DEFAULT ''")
        con.commit()
    except sqlite3.OperationalError:
        pass # Las columnas ya existen
        
    try:
        cur.execute("ALTER TABLE tickets ADD COLUMN archivos_adjuntos TEXT DEFAULT ''")
        cur.execute("ALTER TABLE tickets ADD COLUMN impacto TEXT DEFAULT ''")
        con.commit()
    except sqlite3.OperationalError:
        pass
    con.close()

asegurar_columnas_bitacora()

st.set_page_config(page_title="AUGMENTIQ AOC", page_icon="⚙️", layout="wide")

if 'bandeja_hitl' not in st.session_state:
    st.session_state['bandeja_hitl'] = []
if 'historial_pmo' not in st.session_state:
    st.session_state['historial_pmo'] = []

# --- Esquemas Estructurados (Pydantic) ---
class AnalisisDispatcherAgentes(BaseModel):
    tipo_evaluado_ia: str = Field(description="Clasificación SofOS: 'Incidente' (va a EN PROCESO) o 'Requerimiento' (va a ESTIMANDO)")
    nivel_inconsistencia: str = Field(description="Bajo, Medio, o Alto")
    requiere_hitl: bool = Field(description="True para control humano preventivo si hay inconsistencias o falta de SLA")
    estimacion_horas: float = Field(description="Estimación inicial en horas sugerida por la IA para resolver este ticket.")
    plan_accion_rag: str = Field(description="Resumen técnico de la solución sugerida por la base de conocimientos")
    id_consultor_ideal: str = Field(description="ID del consultor elegido (CONS-01, CONS-02 o CONS-03)")
    justificacion_asignacion: str = Field(description="Explicación precisa de por qué se eligió a este consultor")
    mensaje_para_hitl: str = Field(description="Mensaje estructurado con viñetas del Agente al Supervisor")
    mensaje_interno_consultor: str = Field(description="Borrador para el consultor funcional")
    draft_mensaje_cliente: str = Field(description="Borrador para el cliente. Si falta información, pide disculpas, solicita los datos y avisa explícitamente que el SLA queda en pausa.")

class EvaluacionTracker(BaseModel):
    extraccion_completa: bool = Field(description="True si el texto incluye acción, tiempo y estado final.")
    pregunta_seguimiento: str = Field(description="Pregunta amigable pidiendo el dato faltante.")
    horas_invertidas: float = Field(description="Horas dedicadas. 0.0 si no se menciona.")
    estado_ticket: str = Field(description="'EN PROCESO', 'RESUELTO', o 'BLOQUEADO'.")
    resumen_tecnico: str = Field(description="Resumen formal de la acción.")

class EvaluacionRFI(BaseModel):
    respuesta_aceptable: bool = Field(description="True si el cliente entregó información útil o un avance. False si es evasivo, texto basura o irrelevante.")
    mensaje_cliente_ia: str = Field(description="Si es False, explica qué falta y advierte que el SLA sigue pausado. Si es True, agradece e indica que el equipo retoma el caso.")
    resumen_interno: str = Field(description="Resumen para el equipo de soporte sobre lo que el cliente intentó enviar.")

# --- Funciones de Base de Datos y Finanzas ---
def extraer_tabla(query):
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, con)
    except:
        df = pd.DataFrame()
    con.close()
    return df

def ejecutar_sql(query, params=()):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(query, params)
    con.commit()
    con.close()

def calcular_costo_claude(tokens_in, tokens_out):
    return (tokens_in * 3.0 / 1000000) + (tokens_out * 15.0 / 1000000)

def calcular_estado_sla(fecha_str, estado):
    try:
        fecha_inicio = datetime.datetime.strptime(fecha_str, "%Y-%m-%d %H:%M:%S")
        horas_pasadas = (datetime.datetime.utcnow() - fecha_inicio).total_seconds() / 3600.0
        
        if estado == 'PENDIENTE CLIENTE':
            return "⏸️ SLA Pausado"
        elif estado in ['RESUELTO', 'CANCELADO']:
            return "⏹️ SLA Detenido"
        else:
            if horas_pasadas <= 4:
                return f"🟢 En Tiempo ({horas_pasadas:.1f}h)"
            elif horas_pasadas <= 8:
                return f"🟡 En Riesgo ({horas_pasadas:.1f}h)"
            else:
                return f"🔴 Incumplido ({horas_pasadas:.1f}h)"
    except:
        return "N/A"

def calcular_salud_presupuesto(estimadas, consumidas):
    if pd.isna(estimadas) or estimadas <= 0: return "N/A"
    pct = (consumidas / estimadas) * 100
    if pct < 70: return f"🟢 {pct:.0f}% ({consumidas:.1f}/{estimadas:.1f}h)" 
    elif pct <= 100: return f"🟡 {pct:.0f}% ({consumidas:.1f}/{estimadas:.1f}h)"
    else: return f"🔴 {pct:.0f}% Desviado (+{consumidas-estimadas:.1f}h)"

def procesar_adjuntos_multimodal(archivos, prefijo_id):
    """Guarda físicamente los archivos y los convierte a Base64/Texto para que Claude los analice"""
    lista_nombres = []
    contenido_ia = []
    if not archivos: 
        return lista_nombres, contenido_ia
    
    for arch in archivos:
        nombre = arch.name
        lista_nombres.append(nombre)
        bytes_data = arch.getvalue()
        mime = arch.type
        
        # Guardar localmente
        ruta_guardado = os.path.join(ADJUNTOS_DIR, f"{prefijo_id}_{int(time.time())}_{nombre}")
        with open(ruta_guardado, "wb") as f: 
            f.write(bytes_data)
        
        # Preparar para Visión IA o Análisis de Texto
        if mime in ['image/jpeg', 'image/png', 'image/gif', 'image/webp']:
            b64_data = base64.b64encode(bytes_data).decode('utf-8')
            contenido_ia.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64_data}})
            contenido_ia.append({"type": "text", "text": f"(La imagen anterior es el archivo adjunto: {nombre})"})
        elif mime in ['text/plain', 'text/csv']:
            contenido_ia.append({"type": "text", "text": f"\n--- Contenido adjunto ({nombre}) ---\n{bytes_data.decode('utf-8', errors='ignore')}"})
        else:
            contenido_ia.append({"type": "text", "text": f"\n[Nota al Agente IA: El usuario adjuntó un documento complejo ({nombre}). Se ha guardado en el servidor para revisión manual del consultor.]"})
    
    return lista_nombres, contenido_ia

# --- Enrutadores de Inteligencia Artificial ---
def llamar_agente(contenido_mensaje, esquema, max_intentos=3):
    for intento in range(max_intentos):
        try:
            esquema_json = json.dumps(esquema.model_json_schema())
            mensaje_sistema = (
                "Eres un Agente experto en SAP para el Service Desk de AUGMENTIQ. "
                f"Devuelve ÚNICAMENTE un JSON válido que cumpla estrictamente con este esquema: {esquema_json}."
            )
            respuesta = cliente_claude.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=4096,
                system=mensaje_sistema,
                messages=[{"role": "user", "content": contenido_mensaje}]
            )
            texto_limpio = respuesta.content[-1].text.replace("```json", "").replace("```", "").strip()
            return esquema.model_validate_json(texto_limpio), respuesta.usage.input_tokens, respuesta.usage.output_tokens
        except Exception as e:
            if intento < max_intentos - 1: time.sleep(2 ** intento)
            else:
                st.error(f"Falla crítica en Claude: {str(e)}")
                return None, 0, 0

def llamar_agente_texto(prompt, max_intentos=3, rol="Consultor SAP"):
    if rol == "PMO":
        mensaje_sistema = (
            "Eres el Agente 5 (PMO Advisor IA) de la Torre de Control de AUGMENTIQ. "
            "Tu objetivo es consolidar la telemetría operativa y dotar a la gerencia de visibilidad absoluta y respuestas estratégicas. "
            "Responde de forma directa, analítica y orientada a la rentabilidad corporativa."
        )
    else:
        mensaje_sistema = (
            "Eres un consultor experto en SAP de AUGMENTIQ. Proporciona la respuesta de forma directa y profesional. "
            "NO utilices encabezados Markdown grandes (# o ##). Usa formato de texto plano."
        )
        
    for intento in range(max_intentos):
        try:
            respuesta = cliente_claude.messages.create(
                model="claude-sonnet-5-5",
                max_tokens=4096,
                system=mensaje_sistema,
                messages=[{"role": "user", "content": prompt}]
            )
            return respuesta.content[-1].text, respuesta.usage.input_tokens, respuesta.usage.output_tokens
        except Exception as e:
            if intento < max_intentos - 1: time.sleep(2 ** intento)
            else: return f"Falla en la API: {str(e)}", 0, 0

def obtener_embedding(texto, max_intentos=3):
    for intento in range(max_intentos):
        try:
            return cliente_voyage.embed([texto], model="voyage-3-lite").embeddings[0]
        except Exception as e:
            if intento < max_intentos - 1: time.sleep(2 ** intento)
            else: return None

def calcular_similitud(v1, v2):
    if len(v1) != len(v2): return 0
    return sum(a * b for a, b in zip(v1, v2)) / (math.sqrt(sum(a * a for a in v1)) * math.sqrt(sum(b * b for b in v2)))

# --- PLAYBOOK DINÁMICO (MODAL) ---
@st.dialog("📖 Playbook y Centro de Ayuda (AUGMENTIQ)", width="large")
def mostrar_playbook():
    st.markdown("### Centro de Conocimiento ITSM")
    st.info("Utilice el buscador para filtrar reglas de negocio, definiciones y políticas operativas del Service Desk.")
    
    termino = st.text_input("🔍 Buscar término (ej. SLA, RFI, PMO, RAG)...", "")
    
    playbook_data = {
        "1. Portal Cliente y Reglas de SLA": "El SLA (Service Level Agreement) define los tiempos máximos de atención. Estándares: <4h (Verde), 4-8h (Amarillo), >8h (Rojo/Incumplido). Si un ticket carece de información técnica, el Supervisor ejecuta un RFI (Request For Information), lo que pausa el reloj del SLA automáticamente hasta que el cliente provea los datos por el chat.",
        "2. Control de Gestión y Presupuestos (Supervisor)": "El Supervisor actúa como 'Human in the Loop' (HITL). Analiza y aprueba la estimación inicial de horas sugerida por la IA, estableciendo el Presupuesto del Ticket. Si el cliente agotó sus horas contratadas (Burn Rate superado), el sistema genera una alerta de bloqueo comercial preventivo.",
        "3. Límite WIP y Espacio del Consultor": "Se aplican metodologías ágiles controlando el WIP (Work In Progress). Cada consultor tiene un límite máximo de tickets en paralelo para evitar saturación. Los tickets devueltos al cliente (RFI) liberan WIP temporalmente. El Tracker IA procesa los reportes del consultor e imputa las horas consumidas al ticket automáticamente.",
        "4. IA Interceptora de RFI": "Módulo de protección de SLA. Cuando un ticket está en pausa esperando al cliente, y este responde, la IA intercepta la respuesta. Si el texto es evasivo (ej. 'no sé'), la IA rechaza la entrega, orienta al usuario y mantiene el SLA pausado. Si aporta valor, reactiva el reloj y alerta al equipo.",
        "5. Torre de Control PMO y Burn Rate": "Dashboard ejecutivo. Monitorea el 'Burn Rate' (velocidad de consumo de bolsas de horas) alertando riesgos de rentabilidad. Evalúa la carga del equipo (WIP Limits), el presupuesto de inversión en IA (tokens) y el promedio global de satisfacción del cliente (CSAT).",
        "6. Motor RAG (Base de Conocimiento)": "Uso de IA Vectorial (Retrieval-Augmented Generation). Convierte los nuevos tickets en vectores matemáticos y busca coincidencias semánticas en la base de datos de casos históricos (SQLite) para inyectar automáticamente un plan de acción previo al despacho."
    }
    
    for titulo, contenido in playbook_data.items():
        if termino.lower() in titulo.lower() or termino.lower() in contenido.lower():
            with st.expander(titulo, expanded=True if termino else False):
                st.write(contenido)

# --- Navegación Principal Organizada por Roles ---
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_container_width=True)
else:
    st.sidebar.title("AUGMENTIQ AOC")

menu = st.sidebar.radio("Navegación por Rol", [
    "📥 Portal Cliente (Usuarios)",
    "🛡️ Control de Gestión (Supervisor)",
    "💬 Inbox (Supervisor)",
    "📡 Monitor Operativo (Supervisor)",
    "💬 Espacio de Trabajo (Consultores)",
    "📈 Torre de Control PMO (Gerencia)",
    "⚙️ Configuración Maestra (Admin)"
], help="Simula los permisos de acceso de cada rol (RBAC).")

# --- SELLO COMERCIAL Y BOTÓN DEL PLAYBOOK ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 AUGMENTIQ AOC")
st.sidebar.caption("Agentic Operations Center<br>Elevando el estándar AMS corporativo mediante IA autónoma y gobierno ITIL v4.", unsafe_allow_html=True)

if st.sidebar.button("📖 Abrir Playbook ITSM", use_container_width=True):
    mostrar_playbook()

# --- 1. PORTAL CLIENTE ---
if menu == "📥 Portal Cliente (Usuarios)":
    st.title("Portal de Soporte ITSM", help="Canal de entrada omnicanal basado en historias de usuario.")
    
    df_clientes = extraer_tabla("SELECT id_cliente, nombre_empresa FROM clientes")
    if not df_clientes.empty:
        cliente_sel = st.selectbox("Identifícate (Cliente / Usuario Clave):", df_clientes['nombre_empresa'].tolist())
        id_cliente_sel = df_clientes.loc[df_clientes['nombre_empresa'] == cliente_sel, 'id_cliente'].values[0]
        
        tab_nuevo, tab_seguimiento, tab_email = st.tabs(["📝 Nueva Solicitud", "💬 Seguimiento de Casos y Calidad", "📧 Recepción Automática (Email)"])
        
        with tab_nuevo:
            st.info("💡 Completar este formulario estructurado (aprox. 2 min) permite a nuestro Agente IA analizar y procesar tu requerimiento sin iteraciones.")
            
            with st.container(border=True):
                st.markdown("#### 1️⃣ Información General (Núcleo)")
                col_tipo, col_servicio = st.columns(2)
                tipo_solicitud = col_tipo.selectbox(
                    "Tipo de Solicitud", 
                    ["Seleccionar...", "Incidente (algo dejó de funcionar)", "Solicitud de Servicio (necesito algo nuevo)", "Cambio o mejora", "Consulta"]
                )
                
                servicio_afectado = col_servicio.selectbox(
                    "Servicio Afectado", 
                    ["ERP Financiero (FI/CO)", "Cadena de Suministro (MM/SD/LE)", "Desarrollo y Basis (ABAP/Seguridad)", "Recursos Humanos (HCM)", "Otro"]
                )
                
                resumen_ticket = st.text_input("Resumen (Título corto):", placeholder="Ej. No puedo generar el reporte de facturación.")
                
                col_imp, col_urg = st.columns(2)
                impacto_sel = col_imp.selectbox("Impacto:", ["Solo yo", "Mi equipo / Mi área", "Toda la organización o clientes"])
                urgencia_sel = col_urg.selectbox("Urgencia:", ["Puedo esperar", "Me limita parcialmente", "No puedo trabajar", "Hay fecha límite"])
                
                contacto_alterno = st.text_input("Contacto Alterno (Opcional):", placeholder="Nombre, correo o teléfono de otra persona de contacto en caso de ausencia.")

            if tipo_solicitud != "Seleccionar...":
                with st.container(border=True):
                    st.markdown("#### 2️⃣ Historia de Usuario")
                    
                    if tipo_solicitud == "Consulta":
                        st.markdown("Por favor describe tu duda detalladamente:")
                        descripcion_consulta = st.text_area("Descripción de la Consulta", placeholder="Ej. Quisiera saber cómo puedo descargar el manual de...")
                    else:
                        st.markdown("Ayúdanos a entender el contexto completando las siguientes frases (Formato Agile):")
                        col_hist_1, col_hist_2 = st.columns(2)
                        
                        necesidad = col_hist_1.text_area("¿Qué necesitas lograr? (Quiero poder...)", placeholder="Ej. ...exportar el reporte mensual a Excel.")
                        para_que = col_hist_2.text_area("¿Para qué lo necesitas? (Para...)", placeholder="Ej. ...enviarlo a gerencia antes del cierre contable.")
                        
                        situacion_actual = st.text_area("¿Qué pasa hoy? (Contexto/Situación actual)", placeholder="Ej. Actualmente al intentar exportarlo el sistema se queda colgado.")
                        
                        # --- CAMPOS CONDICIONALES ---
                        if tipo_solicitud == "Incidente (algo dejó de funcionar)":
                            st.markdown("##### Detalles del Incidente")
                            col_inc1, col_inc2 = st.columns(2)
                            cuando_empezo = col_inc1.text_input("¿Cuándo empezó?", placeholder="Ej. Ayer en la tarde")
                            es_recurrente = col_inc2.selectbox("¿Es la primera vez?", ["Sí, primera vez", "No, ya había pasado"])
                            mensaje_error = st.text_area("Mensaje de error exacto (si lo hay):", placeholder="Ej. Error 403 o Dump ABAP KD503")
                            
                        elif tipo_solicitud == "Solicitud de Servicio (necesito algo nuevo)":
                            st.markdown("##### Detalles de la Solicitud")
                            col_sol1, col_sol2 = st.columns(2)
                            para_quien = col_sol1.selectbox("¿Para quién es?", ["Para mí", "Para otra persona de mi equipo"])
                            aprobador = col_sol2.text_input("Aprobador (Jefe directo):", placeholder="Correo o nombre de quien aprueba el acceso/compra.")
                            
                        elif tipo_solicitud == "Cambio o mejora":
                            st.markdown("##### Detalles del Cambio")
                            beneficio = st.text_area("Beneficio esperado:", placeholder="Ej. Reducción de errores manuales en un 20%")
                            riesgo = st.text_area("Riesgo si no se hace (Opcional):", placeholder="Ej. Retrasos en las entregas a clientes finales.")

                    st.markdown("#### 3️⃣ Criterios de Aceptación y Evidencias")
                    criterios_aceptacion = st.text_area("¿Cómo sabremos que está resuelto? (Criterio de éxito)", placeholder="Ej. Cuando pueda descargar el reporte en formato .xlsx sin que la pantalla se congele.")
                    
                    archivos_subidos = st.file_uploader("📎 Adjuntar evidencias (Capturas, PDF, Excel, Logs)", accept_multiple_files=True)
                    st.caption("💡 *Nota: Después de enviar tu solicitud, el sistema subirá los archivos automáticamente.*")
            
                if st.button("Enviar Solicitud al Service Desk", type="primary"):
                    if not resumen_ticket:
                        st.error("El campo 'Resumen' es obligatorio.")
                    elif tipo_solicitud == "Consulta" and not descripcion_consulta:
                        st.error("Por favor ingresa la descripción de tu consulta.")
                    elif tipo_solicitud != "Consulta" and (not necesidad or not situacion_actual):
                        st.error("Por favor completa los campos '¿Qué necesitas lograr?' y '¿Qué pasa hoy?' para armar la historia de usuario.")
                    else:
                        with st.spinner("El Analista IA de Primera Línea está evaluando tu solicitud y las evidencias adjuntas..."):
                            id_temp = f"TMP-{int(time.time())}"
                            fecha_hora_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # --- ARMADO DE LA HISTORIA ESTRUCTURADA PARA LA IA Y EL CONSULTOR ---
                            if tipo_solicitud == "Consulta":
                                historia_completa = f"TIPO: {tipo_solicitud} | SERVICIO: {servicio_afectado}\nTÍTULO: {resumen_ticket}\nDESCRIPCIÓN: {descripcion_consulta}\nIMPACTO: {impacto_sel} | URGENCIA: {urgencia_sel}\nCONTACTO ALTERNO: {contacto_alterno}"
                            else:
                                historia_completa = f"TIPO: {tipo_solicitud} | SERVICIO: {servicio_afectado}\nTÍTULO: {resumen_ticket}\n"
                                historia_completa += f"\nHISTORIA DE USUARIO:\nComo usuario de {cliente_sel}, quiero poder {necesidad}, para {para_que}.\n"
                                historia_completa += f"\nSITUACIÓN ACTUAL:\n{situacion_actual}\n"
                                historia_completa += f"\nCRITERIOS DE ACEPTACIÓN:\n- {criterios_aceptacion}\n"
                                
                                if tipo_solicitud == "Incidente (algo dejó de funcionar)":
                                    historia_completa += f"\nDETALLES EXTRA: Empezó: {cuando_empezo} | Recurrente: {es_recurrente}\nError: {mensaje_error}\n"
                                elif tipo_solicitud == "Solicitud de Servicio (necesito algo nuevo)":
                                    historia_completa += f"\nDETALLES EXTRA: Solicitante final: {para_quien} | Aprobador: {aprobador}\n"
                                elif tipo_solicitud == "Cambio o mejora":
                                    historia_completa += f"\nDETALLES EXTRA: Beneficio: {beneficio} | Riesgo: {riesgo}\n"
                                    
                                historia_completa += f"\nIMPACTO: {impacto_sel} | URGENCIA: {urgencia_sel}\nCONTACTO ALTERNO: {contacto_alterno}"
                            
                            vector_busqueda = obtener_embedding(historia_completa)
                            mejor_solucion_rag = "No se encontraron casos históricos similares."
                            if vector_busqueda:
                                con = sqlite3.connect(DB_PATH)
                                try:
                                    filas = con.execute("SELECT solucion, vector_embedding FROM BaseConocimiento").fetchall()
                                    mejor_similitud = 0
                                    for fila in filas:
                                        similitud = calcular_similitud(vector_busqueda, json.loads(fila[1]))
                                        if similitud > 0.40 and similitud > mejor_similitud:
                                            mejor_similitud, mejor_solucion_rag = similitud, fila[0]
                                except: pass
                                con.close()

                            con = sqlite3.connect(DB_PATH)
                            datos_cliente = con.execute(f"SELECT nombre_empresa, horas_contratadas, horas_consumidas FROM clientes WHERE id_cliente='{id_cliente_sel}'").fetchone()
                            datos_consultores = con.execute("SELECT id_consultor, nombre, especialidad_sap, tickets_activos, limite_wip, estado FROM consultores").fetchall()
                            con.close()
                            
                            horas_contratadas = float(datos_cliente[1])
                            horas_consumidas = float(datos_cliente[2])
                            if horas_consumidas >= horas_contratadas:
                                alerta_msg = f"🚨 ALERTA DE SLA: El cliente {cliente_sel} ha reportado una solicitud URGENCIA: {urgencia_sel} pero tiene su bolsa AGOTADA. Requiere autorización."
                            else:
                                alerta_msg = f"ℹ️ NUEVO CASO: Solicitud de {cliente_sel} (Urgencia: {urgencia_sel}) recibida ({id_temp})."
                            
                            ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                                         (id_temp, "Dispatcher IA", "Supervisor", alerta_msg, True))

                            # --- PREPARACIÓN MULTIMODAL PARA CLAUDE ---
                            nombres_archivos, contenido_ia_extra = procesar_adjuntos_multimodal(archivos_subidos, id_temp)
                            texto_base = f"""Nueva solicitud de {datos_cliente[0]}.
Historia de Usuario:
{historia_completa}
Solución Histórica RAG (Si la hay): {mejor_solucion_rag}
Horas Contratadas: {horas_contratadas} | Horas Consumidas: {horas_consumidas}
Directorio Consultores (Activos/WIP Límite): {datos_consultores}
Analiza y genera el JSON. Justifica la asignación evaluando el WIP y estima las horas necesarias. Redacta el mensaje interno y el borrador al cliente (si falta info técnica como transacciones o capturas, indícalo con empatía e incluye la advertencia de que el SLA entra en pausa)."""
                            
                            prompt_multimodal = [{"type": "text", "text": texto_base}] + contenido_ia_extra
                            
                            analisis, t_in, t_out = llamar_agente(prompt_multimodal, AnalisisDispatcherAgentes)
                            if analisis:
                                costo = calcular_costo_claude(t_in, t_out)
                                st.session_state['bandeja_hitl'].append({
                                    'id_temp': id_temp,
                                    'fecha_hora': fecha_hora_actual,
                                    'urgencia': urgencia_sel,
                                    'impacto': impacto_sel,
                                    'id_cliente': id_cliente_sel,
                                    'nombre_cliente': cliente_sel,
                                    'solicitud': historia_completa,
                                    'archivos': ", ".join(nombres_archivos),
                                    'analisis': analisis,
                                    'draft_interno_actual': analisis.mensaje_interno_consultor,
                                    'consultor_actual': analisis.id_consultor_ideal,
                                    'draft_cliente_actual': analisis.draft_mensaje_cliente,
                                    'tokens': t_in + t_out,
                                    'costo': costo
                                })
                                st.success("✅ Solicitud recibida. Las evidencias fueron procesadas y el ticket está en fase NUEVO.")

        with tab_seguimiento:
            df_tickets_cli = extraer_tabla(f"SELECT id_ticket, estado_ticket, calificacion_cliente FROM tickets WHERE id_cliente = '{id_cliente_sel}' ORDER BY fecha_creacion DESC")
            if not df_tickets_cli.empty:
                ticket_cli_sel = st.selectbox("Selecciona un ticket para consultar o valorar:", df_tickets_cli['id_ticket'].tolist())
                estado_actual = df_tickets_cli.loc[df_tickets_cli['id_ticket'] == ticket_cli_sel, 'estado_ticket'].values[0]
                calificacion = df_tickets_cli.loc[df_tickets_cli['id_ticket'] == ticket_cli_sel, 'calificacion_cliente'].values[0]
                
                st.write("---")
                if estado_actual == "RESUELTO":
                    if calificacion == 0:
                        st.info("🎯 **¡Ticket Resuelto!** Ayúdanos a mejorar valorando el servicio recibido.")
                        estrellas = st.feedback("stars") # Interfaz de valoración CSAT
                        if estrellas is not None:
                            ejecutar_sql("UPDATE tickets SET calificacion_cliente = ? WHERE id_ticket = ?", (estrellas + 1, ticket_cli_sel))
                            st.success("¡Gracias por tu feedback!")
                            st.rerun()
                    else:
                        st.success(f"Valoración del ticket: {'⭐' * calificacion}")
                
                chat_cli_df = extraer_tabla(f"SELECT aprobado_por as remitente, cuerpo_mensaje as mensaje, fecha_creacion FROM comunicaciones_cliente WHERE id_ticket = '{ticket_cli_sel}' ORDER BY fecha_creacion ASC")
                for index, row in chat_cli_df.iterrows():
                    is_cliente = row['remitente'] == 'Cliente'
                    with st.chat_message("user" if is_cliente else "assistant"):
                        st.markdown(f"**{row['remitente']}** ({row['fecha_creacion'][:16]})\n\n{row['mensaje']}")
                        
                archivos_chat_cli = st.file_uploader("📎 Anexar evidencias a tu respuesta (Opcional)", accept_multiple_files=True, key=f"up_cli_{ticket_cli_sel}")        
                st.caption("💡 *Nota: Después de enviar tu mensaje, presiona la 'X' en el archivo adjunto para limpiar la bandeja.*")
                resp_cli = st.chat_input("Escribe tu consulta o proporciona la información solicitada...")
                
                if resp_cli:
                    nombres_archivos, contenido_ia_extra = procesar_adjuntos_multimodal(archivos_chat_cli, ticket_cli_sel)
                    mensaje_final = resp_cli
                    if nombres_archivos:
                        mensaje_final += f"\n\n📎 *Archivos anexos: {', '.join(nombres_archivos)}*"
                        
                    ejecutar_sql("INSERT INTO comunicaciones_cliente (id_ticket, estado_mensaje, cuerpo_mensaje, aprobado_por) VALUES (?, 'Enviado', ?, 'Cliente')", (ticket_cli_sel, mensaje_final))
                    
                    if estado_actual == "PENDIENTE CLIENTE":
                        with st.spinner("La IA Interceptora está evaluando tu información y evidencias visuales..."):
                            historial = "\n".join([f"{r['remitente']}: {r['mensaje']}" for _, r in chat_cli_df.iterrows()])
                            prompt_base = f"Historial del ticket:\n{historial}\n\nNueva respuesta del usuario: '{mensaje_final}'.\nEvalúa si la respuesta aporta la información solicitada o al menos un avance técnico válido."
                            
                            prompt_multimodal = [{"type": "text", "text": prompt_base}] + contenido_ia_extra
                            eval_rfi, t_in, t_out = llamar_agente(prompt_multimodal, EvaluacionRFI)
                            
                            if eval_rfi:
                                c_ia = calcular_costo_claude(t_in, t_out)
                                ejecutar_sql("UPDATE tickets SET tokens_usados = tokens_usados + ?, costo_ia = costo_ia + ? WHERE id_ticket = ?", (t_in + t_out, c_ia, ticket_cli_sel))
                                
                                ejecutar_sql("INSERT INTO comunicaciones_cliente (id_ticket, estado_mensaje, cuerpo_mensaje, aprobado_por) VALUES (?, 'Enviado', ?, 'Agente IA')", (ticket_cli_sel, eval_rfi.mensaje_cliente_ia))
                                
                                if eval_rfi.respuesta_aceptable:
                                    ejecutar_sql("UPDATE tickets SET estado_ticket = 'EN PROCESO' WHERE id_ticket = ?", (ticket_cli_sel,))
                                    ejecutar_sql("UPDATE consultores SET tickets_activos = tickets_activos + 1 WHERE id_consultor = (SELECT id_consultor_asignado FROM tickets WHERE id_ticket = ?)", (ticket_cli_sel,))
                                    ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                                                 (ticket_cli_sel, "Agente IA", "Supervisor", f"✅ RFI Aceptado (Evidencia procesada): {eval_rfi.resumen_interno}. El ticket regresó a EN PROCESO (SLA Reactivado).", True))
                                else:
                                    ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                                                 (ticket_cli_sel, "Agente IA", "Supervisor", f"⚠️ RFI Rechazado al cliente: {eval_rfi.resumen_interno}. El SLA se mantiene en pausa.", False))
                    else:
                        ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                                     (ticket_cli_sel, "Cliente", "Supervisor", f"🚨 El cliente ha enviado un nuevo mensaje en el ticket {ticket_cli_sel}:\n'{mensaje_final}'", True))
                    st.rerun()
            else:
                st.info("No tienes tickets activos o históricos en el sistema.")
                
        with tab_email:
            st.info("🤖 **Omnicanalidad (Email-to-Ticket):** Simulación de ingesta automática de correos.")
            
            with st.expander("⚙️ ¿Cómo funciona este proceso en la vida real (Producción)?", expanded=False):
                st.markdown("""
                **Arquitectura del Proceso Automático (Cron Job):**
                * **Monitoreo Continuo:** En un entorno de producción, no es necesario hacer clic en un botón. Un microservicio en la nube (ej. Google Cloud Functions o AWS Lambda) se ejecuta silenciosamente cada *X* minutos.
                * **Conexión Segura al Buzón:** El servicio se conecta directamente al buzón oficial de la empresa (ej. `soporte@sofos.com`) utilizando protocolos de integración modernos (API de Gmail, Microsoft Graph o IMAP).
                * **Gestión de Credenciales:** La dirección de correo a leer y sus contraseñas/tokens no están quemadas en el código; se configuran desde variables de entorno seguras (`.env`) o bóvedas como AWS Secret Manager.
                * **Ruteo Inteligente:** Cuando llega un correo nuevo, la IA lo lee, identifica al cliente por su dominio (`@agroexportadora.cr`), estructura la historia de usuario y lo inserta en la base de datos automáticamente.
                
                *Para efectos de esta demostración, el botón inferior simula el instante exacto en que ese proceso en background detecta y lee un nuevo correo.*
                """)
                
            st.markdown("#### 📥 Bandeja de Entrada (Simulación del último correo recibido)")
            with st.container(border=True):
                st.markdown("**De:** j.perez@agroexportadora.cr")
                st.markdown("**Asunto:** URGENTE - F.05 arrojando Dump ABAP")
                st.markdown("**Fecha:** Hace 3 minutos")
                st.write("---")
                cuerpo_email = "Hola equipo de soporte,\n\nDesde esta mañana estamos intentando ejecutar la transacción F.05 para la valoración de moneda extranjera y el sistema nos arroja un 'Dump de memoria ABAP'. Necesitamos que lo revisen con urgencia extrema porque hoy tenemos el cierre contable mensual y no podemos emitir los estados financieros.\n\nPor favor su apoyo inmediato.\nSaludos, J. Perez."
                st.write(cuerpo_email)
            
            if st.button("🔄 Procesar Correo con IA (Convertir a Ticket)", type="primary"):
                with st.spinner("El Agente Dispatcher IA está leyendo el correo, identificando al cliente y estructurando la historia de usuario..."):
                    id_temp = f"TMP-{int(time.time())}"
                    fecha_hora_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    id_cli_email = 'CLI-01'
                    cliente_email = 'Agro Exportadora CR'
                    
                    historia_estructurada = f"TIPO: Incidente (algo dejó de funcionar) | SERVICIO: ERP Financiero (FI/CO)\nTÍTULO: Dump de memoria ABAP en transacción F.05\n\nHISTORIA DE USUARIO:\nComo usuario de {cliente_email}, quiero poder ejecutar la transacción F.05, para realizar la valoración de moneda extranjera del cierre contable.\n\nSITUACIÓN ACTUAL:\nEl sistema arroja un Dump de memoria ABAP.\n\nCRITERIOS DE ACEPTACIÓN:\n- Que la transacción F.05 se ejecute sin errores de memoria.\n\nDETALLES EXTRA: Empezó: Esta mañana | Error: Dump de memoria ABAP\nIMPACTO: Toda la organización o clientes | URGENCIA: No puedo trabajar\nCONTACTO ALTERNO: j.perez@agroexportadora.cr"
                    
                    vector_busqueda = obtener_embedding(historia_estructurada)
                    mejor_solucion_rag = "No se encontraron casos históricos similares."
                    if vector_busqueda:
                        con = sqlite3.connect(DB_PATH)
                        try:
                            filas = con.execute("SELECT solucion, vector_embedding FROM BaseConocimiento").fetchall()
                            mejor_similitud = 0
                            for fila in filas:
                                similitud = calcular_similitud(vector_busqueda, json.loads(fila[1]))
                                if similitud > 0.40 and similitud > mejor_similitud:
                                    mejor_similitud, mejor_solucion_rag = similitud, fila[0]
                        except: pass
                        con.close()

                    con = sqlite3.connect(DB_PATH)
                    datos_cliente = con.execute(f"SELECT nombre_empresa, horas_contratadas, horas_consumidas FROM clientes WHERE id_cliente='{id_cli_email}'").fetchone()
                    datos_consultores = con.execute("SELECT id_consultor, nombre, especialidad_sap, tickets_activos, limite_wip, estado FROM consultores").fetchall()
                    con.close()
                    
                    horas_contratadas = float(datos_cliente[1])
                    horas_consumidas = float(datos_cliente[2])
                    
                    alerta_msg = f"📧 INGESTA AUTOMÁTICA: Solicitud de {cliente_email} procesada desde email ({id_temp})."
                    ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)", (id_temp, "Dispatcher IA", "Supervisor", alerta_msg, True))

                    texto_base = f"""Se recibió el siguiente correo del cliente {datos_cliente[0]}:\n"{cuerpo_email}"\n
La IA ya lo estructuró de la siguiente forma:
{historia_estructurada}
Solución Histórica RAG: {mejor_solucion_rag}
Horas Contratadas: {horas_contratadas} | Horas Consumidas: {horas_consumidas}
Directorio Consultores: {datos_consultores}
Analiza y genera el JSON asignando al mejor consultor."""
                    
                    analisis, t_in, t_out = llamar_agente([{"type": "text", "text": texto_base}], AnalisisDispatcherAgentes)
                    
                    if analisis:
                        costo = calcular_costo_claude(t_in, t_out)
                        st.session_state['bandeja_hitl'].append({
                            'id_temp': id_temp,
                            'fecha_hora': fecha_hora_actual,
                            'urgencia': "No puedo trabajar",
                            'impacto': "Toda la organización o clientes",
                            'id_cliente': id_cli_email,
                            'nombre_cliente': cliente_email,
                            'solicitud': historia_estructurada,
                            'archivos': "",
                            'analisis': analisis,
                            'draft_interno_actual': analisis.mensaje_interno_consultor,
                            'consultor_actual': analisis.id_consultor_ideal,
                            'draft_cliente_actual': analisis.draft_mensaje_cliente,
                            'tokens': t_in + t_out,
                            'costo': costo
                        })
                        st.success("✅ ¡Correo procesado exitosamente! El ticket estructurado ha sido enviado a la Bandeja de Aprobación del Supervisor.")

    else: st.warning("No hay clientes en la base de datos.")

# --- 2. CONTROL DE GESTIÓN (HITL) ---
elif menu == "🛡️ Control de Gestión (Supervisor)":
    st.title("Bandeja de Aprobación (Supervisor)", help="Aplica políticas ITIL: Aprueba para asignar, RFI para pedir info al cliente, o Rechaza si no procede.")
    
    cols_kpi = st.columns(3)
    cols_kpi[0].metric("Casos en Cola", len(st.session_state['bandeja_hitl']))
    cols_kpi[1].metric("SLA Activos (Nuevos)", len(st.session_state['bandeja_hitl']))
    
    st.write("---")
    
    if not st.session_state['bandeja_hitl']:
        st.write("La bandeja está limpia. No hay tickets en cola.")
    else:
        df_cons = extraer_tabla("SELECT id_consultor, nombre FROM consultores")
        lista_consultores = df_cons['id_consultor'].tolist()
        
        for idx, caso in enumerate(st.session_state['bandeja_hitl']):
            analisis = caso['analisis']
            icono_riesgo = "🔴" if analisis.nivel_inconsistencia == "Alto" else ("🟡" if analisis.nivel_inconsistencia == "Medio" else "🟢")
            
            urgencia_str = caso.get('urgencia', 'Media').split(' ')[0]
            resumen_tarjeta = f"{icono_riesgo} **{caso['id_temp']}** | 🕒 {caso.get('fecha_hora', '')} | ⚠️ {urgencia_str} | Cliente: {caso['nombre_cliente']} | Tipo: {analisis.tipo_evaluado_ia}"
            
            with st.expander(resumen_tarjeta):
                st.markdown(f"**Historia de Usuario Registrada:**\n\n{caso['solicitud']}")
                if caso.get('archivos'): 
                    st.caption(f"📎 **Evidencias adjuntas:** {caso['archivos']}")
                
                st.info(f"🤖 **Análisis del Agente (IA):**\n\n{analisis.mensaje_para_hitl}")
                
                col_asig, col_com = st.columns(2)
                
                with col_asig:
                    st.write("#### 👨‍💻 Asignación y Presupuesto")
                    st.write(f"**Justificación:** {analisis.justificacion_asignacion}")
                    
                    idx_sugerido = lista_consultores.index(caso['consultor_actual']) if caso['consultor_actual'] in lista_consultores else 0
                    consultor_definitivo = st.selectbox("Confirmar/Modificar Asignación:", lista_consultores, index=idx_sugerido, key=f"sel_{caso['id_temp']}")
                    
                    horas_ppto = st.number_input("Presupuesto de Horas (Estimación IA):", value=float(analisis.estimacion_horas), min_value=0.5, step=0.5, key=f"hrs_{caso['id_temp']}", help="Línea base para medir la desviación del consultor en el Monitor.")
                    
                    if consultor_definitivo != caso['consultor_actual']:
                        with st.spinner("Reescribiendo borrador..."):
                            nuevo_nombre = df_cons.loc[df_cons['id_consultor'] == consultor_definitivo, 'nombre'].values[0]
                            prompt_reescribir = f"Adapta este mensaje: '{caso['draft_interno_actual']}'. Reasigna a {nuevo_nombre} ({consultor_definitivo})."
                            nuevo_draft, t_in, t_out = llamar_agente_texto(prompt_reescribir)
                            caso['tokens'] += (t_in + t_out)
                            caso['costo'] += calcular_costo_claude(t_in, t_out)
                            caso['draft_interno_actual'] = nuevo_draft
                            caso['consultor_actual'] = consultor_definitivo
                            st.rerun()
                            
                    st.write(f"**Plan de Acción (RAG):** {analisis.plan_accion_rag}")
                    
                    st.write("---")
                    st.write("#### 📝 Control de Calidad (Obligatorio para Devolver/Rechazar)")
                    motivo_excepcion = st.text_area("Justificación del Supervisor:", key=f"motivo_{caso['id_temp']}", help="Documenta por qué se detiene el SLA o se rechaza el ticket. Visible en reportes de calidad.")

                with col_com:
                    st.write("#### ✉️ Edición de Borradores")
                    st.write(f"**Flujo SofOS:** {analisis.tipo_evaluado_ia}")
                    texto_cliente_final = st.text_area("Mensaje al Usuario Clave (Cliente):", caso['draft_cliente_actual'], height=250, key=f"txt_cli_{caso['id_temp']}", help="Este texto se enviará al cliente tanto si apruebas, cancelas o devuelves el ticket.")
                    texto_interno_final = st.text_area("Mensaje al Consultor Asignado:", caso['draft_interno_actual'], height=250, key=f"txt_int_{caso['id_temp']}")
                
                st.write("---")
                col_btn1, col_btn2, col_btn3 = st.columns(3)
                
                if col_btn1.button("✅ Aprobar y Asignar", key=f"btn_apr_{caso['id_temp']}", use_container_width=True, help="Inicia el reloj del SLA y asigna al consultor."):
                    id_ticket = caso['id_temp'].replace("TMP", "TKT")
                    estado_inicial = "EN PROCESO" if analisis.tipo_evaluado_ia == "Incidente" else "ESTIMANDO"
                    # Usar el nombre del consultor
                    nombre_consultor = df_cons.loc[df_cons['id_consultor'] == consultor_definitivo, 'nombre'].values[0]
                    
                    ejecutar_sql(
                        "INSERT INTO tickets (id_ticket, id_cliente, urgencia, descripcion_cliente, tipo_evaluado_ia, id_consultor_asignado, estado_ticket, plan_accion_sugerido, tokens_usados, costo_ia, motivo_excepcion, horas_estimadas, archivos_adjuntos, impacto) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (id_ticket, caso['id_cliente'], caso.get('urgencia', ''), caso['solicitud'], analisis.tipo_evaluado_ia, consultor_definitivo, estado_inicial, analisis.plan_accion_rag, caso['tokens'], caso['costo'], motivo_excepcion, horas_ppto, caso.get('archivos', ''), caso.get('impacto', ''))
                    )
                    ejecutar_sql("UPDATE consultores SET tickets_activos = tickets_activos + 1 WHERE id_consultor = ?", (consultor_definitivo,))
                    ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)", (id_ticket, "Supervisor", nombre_consultor, texto_interno_final, True))
                    ejecutar_sql("INSERT INTO comunicaciones_cliente (id_ticket, estado_mensaje, cuerpo_mensaje, aprobado_por) VALUES (?, ?, ?, ?)", (id_ticket, "Enviado", texto_cliente_final, "Supervisor"))
                    
                    st.session_state['bandeja_hitl'] = [c for c in st.session_state['bandeja_hitl'] if c['id_temp'] != caso['id_temp']]
                    st.success(f"Ticket {id_ticket} asignado a {nombre_consultor}.")
                    time.sleep(1)
                    st.rerun()
                
                if col_btn2.button("⚠️ Devolver (RFI - Pausar SLA)", key=f"btn_rfi_{caso['id_temp']}", use_container_width=True, help="Devuelve el ticket por falta de información. Pausa el reloj del SLA."):
                    if not motivo_excepcion:
                        st.error("Debes ingresar una Justificación en la columna izquierda para registrar la pausa del SLA.")
                    else:
                        id_ticket = caso['id_temp'].replace("TMP", "TKT")
                        ejecutar_sql(
                            "INSERT INTO tickets (id_ticket, id_cliente, urgencia, descripcion_cliente, tipo_evaluado_ia, id_consultor_asignado, estado_ticket, plan_accion_sugerido, tokens_usados, costo_ia, motivo_excepcion, horas_estimadas, archivos_adjuntos, impacto) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (id_ticket, caso['id_cliente'], caso.get('urgencia', ''), caso['solicitud'], analisis.tipo_evaluado_ia, consultor_definitivo, "PENDIENTE CLIENTE", analisis.plan_accion_rag, caso['tokens'], caso['costo'], motivo_excepcion, horas_ppto, caso.get('archivos', ''), caso.get('impacto', ''))
                        )
                        ejecutar_sql("INSERT INTO comunicaciones_cliente (id_ticket, estado_mensaje, cuerpo_mensaje, aprobado_por) VALUES (?, ?, ?, ?)", (id_ticket, "Enviado", texto_cliente_final, "Supervisor"))
                        ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)", (id_ticket, "Sistema AOC", "Supervisor", f"SLA Pausado. Motivo: {motivo_excepcion}", False))
                        
                        st.session_state['bandeja_hitl'] = [c for c in st.session_state['bandeja_hitl'] if c['id_temp'] != caso['id_temp']]
                        st.warning(f"Ticket {id_ticket} devuelto al cliente. SLA en pausa.")
                        time.sleep(1)
                        st.rerun()

                if col_btn3.button("❌ Rechazar / Cancelar", key=f"btn_rech_{caso['id_temp']}", use_container_width=True, help="Descarta la solicitud (ej: duplicado)."):
                    if not motivo_excepcion:
                        st.error("Debes ingresar una Justificación en la columna izquierda para cancelar un ticket.")
                    else:
                        id_ticket = caso['id_temp'].replace("TMP", "TKT")
                        ejecutar_sql(
                            "INSERT INTO tickets (id_ticket, id_cliente, urgencia, descripcion_cliente, tipo_evaluado_ia, id_consultor_asignado, estado_ticket, plan_accion_sugerido, tokens_usados, costo_ia, motivo_excepcion, horas_estimadas, archivos_adjuntos, impacto) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (id_ticket, caso['id_cliente'], caso.get('urgencia', ''), caso['solicitud'], analisis.tipo_evaluado_ia, consultor_definitivo, "CANCELADO", analisis.plan_accion_rag, caso['tokens'], caso['costo'], motivo_excepcion, 0, caso.get('archivos', ''), caso.get('impacto', ''))
                        )
                        ejecutar_sql("INSERT INTO comunicaciones_cliente (id_ticket, estado_mensaje, cuerpo_mensaje, aprobado_por) VALUES (?, ?, ?, ?)", (id_ticket, "Enviado", texto_cliente_final, "Supervisor"))
                        ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)", (id_ticket, "Sistema AOC", "Supervisor", f"Ticket Cancelado. Motivo: {motivo_excepcion}", False))
                        
                        st.session_state['bandeja_hitl'] = [c for c in st.session_state['bandeja_hitl'] if c['id_temp'] != caso['id_temp']]
                        st.error(f"Solicitud {id_ticket} cancelada.")
                        time.sleep(1)
                        st.rerun()

# --- 2.5 INBOX SUPERVISOR ---
elif menu == "💬 Inbox (Supervisor)":
    st.title("Inbox del Supervisor de Operaciones")
    
    df_tickets_sup = extraer_tabla("SELECT DISTINCT id_ticket FROM comunicaciones_internas WHERE id_ticket != 'PMO-LOG' ORDER BY id_mensaje DESC")
    
    if not df_tickets_sup.empty:
        ticket_sel = st.selectbox("Selecciona el caso a gestionar/auditar:", df_tickets_sup['id_ticket'].tolist())
        
        canal = st.radio("Seleccione el canal de comunicación:", ["🏢 Chat Interno (Equipo e IA)", "🤝 Chat con Cliente"], horizontal=True)
        st.write("---")
        
        if canal == "🏢 Chat Interno (Equipo e IA)":
            chat_df = extraer_tabla(f"SELECT remitente, destinatario, mensaje, fecha_envio FROM comunicaciones_internas WHERE id_ticket = '{ticket_sel}' ORDER BY fecha_envio ASC")
            for index, row in chat_df.iterrows():
                is_supervisor = row['remitente'] == 'Supervisor'
                with st.chat_message("user" if is_supervisor else "ai"):
                    st.markdown(f"**{row['remitente']}** a **{row['destinatario']}** ({row['fecha_envio'][:16]})\n\n{row['mensaje']}")

            archivos_chat_sup_int = st.file_uploader("📎 Anexar evidencia técnica para el equipo (Opcional)", accept_multiple_files=True, key=f"up_sup_int_{ticket_sel}")
            st.caption("💡 *Nota: Después de enviar tu mensaje, presiona la 'X' en el archivo adjunto para limpiar la bandeja.*")
            respuesta = st.chat_input("Escribe una instrucción para la IA o el equipo...")
            
            if respuesta:
                nombres_archivos, contenido_ia_extra = procesar_adjuntos_multimodal(archivos_chat_sup_int, ticket_sel)
                mensaje_final = respuesta
                if nombres_archivos:
                    mensaje_final += f"\n\n📎 *Archivos anexos: {', '.join(nombres_archivos)}*"
                
                ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                             (ticket_sel, "Supervisor", "Tracker IA", mensaje_final, False))
                with st.spinner("El Agente IA está procesando tu instrucción..."):
                    prompt_base = f"Eres el Agente IA (Tracker IA) de AUGMENTIQ. El supervisor te dice sobre el ticket {ticket_sel}: '{mensaje_final}'. Responde directamente al supervisor."
                    prompt_multimodal = [{"type": "text", "text": prompt_base}] + contenido_ia_extra
                    
                    respuesta_ia, t_in, t_out = llamar_agente_texto(prompt_multimodal)
                    
                    c_ia = calcular_costo_claude(t_in, t_out)
                    ejecutar_sql("UPDATE tickets SET tokens_usados = tokens_usados + ?, costo_ia = costo_ia + ? WHERE id_ticket = ?", (t_in + t_out, c_ia, ticket_sel))
                    
                    ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                                 (ticket_sel, "Tracker IA", "Supervisor", respuesta_ia, False))
                st.rerun()
                
        else:
            chat_cli_df = extraer_tabla(f"SELECT aprobado_por as remitente, cuerpo_mensaje as mensaje, fecha_creacion FROM comunicaciones_cliente WHERE id_ticket = '{ticket_sel}' ORDER BY fecha_creacion ASC")
            for index, row in chat_cli_df.iterrows():
                is_internal = row['remitente'] != 'Cliente'
                with st.chat_message("assistant" if is_internal else "user"):
                    st.markdown(f"**{row['remitente']}** ({row['fecha_creacion'][:16]})\n\n{row['mensaje']}")
            
            archivos_chat_sup_ext = st.file_uploader("📎 Anexar documento o captura para el cliente (Opcional)", accept_multiple_files=True, key=f"up_sup_ext_{ticket_sel}")
            st.caption("💡 *Nota: Después de enviar tu mensaje, presiona la 'X' en el archivo adjunto para limpiar la bandeja.*")        
            resp_to_cli = st.chat_input("Responde directamente al cliente (Visible en el Portal Cliente)...")
            
            if resp_to_cli:
                nombres_archivos, _ = procesar_adjuntos_multimodal(archivos_chat_sup_ext, ticket_sel)
                mensaje_final = resp_to_cli
                if nombres_archivos:
                    mensaje_final += f"\n\n📎 *Archivos anexos: {', '.join(nombres_archivos)}*"
                    
                ejecutar_sql("INSERT INTO comunicaciones_cliente (id_ticket, estado_mensaje, cuerpo_mensaje, aprobado_por) VALUES (?, 'Enviado', ?, 'Supervisor')", (ticket_sel, mensaje_final))
                st.rerun()
    else:
        st.info("No hay notificaciones ni mensajes en el buzón del supervisor.")

# --- 3. MONITOR DE OPERACIONES ---
elif menu == "📡 Monitor Operativo (Supervisor)":
    st.title("Monitor de Trabajo Diario")
    
    query_activos = """
        SELECT
            t.id_ticket,
            c.nombre_empresa as Cliente,
            t.urgencia as Urgencia,
            cons.nombre as Consultor,
            t.estado_ticket as Estado,
            t.fecha_creacion,
            t.horas_estimadas,
            t.horas_consumidas_ticket
        FROM tickets t
        LEFT JOIN clientes c ON t.id_cliente = c.id_cliente
        LEFT JOIN consultores cons ON t.id_consultor_asignado = cons.id_consultor
        ORDER BY t.fecha_creacion DESC
    """
    df_activos = extraer_tabla(query_activos)
    
    if not df_activos.empty:
        df_activos['SLA_Status'] = df_activos.apply(lambda row: calcular_estado_sla(row['fecha_creacion'], row['Estado']), axis=1)
        df_activos['Salud_Ppto'] = df_activos.apply(lambda row: calcular_salud_presupuesto(row['horas_estimadas'], row['horas_consumidas_ticket']), axis=1)
        
        st.subheader("📊 Resumen Operativo")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total de Tickets (BD)", len(df_activos))
        kpi2.metric("En Progreso (WIP)", len(df_activos[df_activos['Estado'].isin(['EN PROCESO', 'ESTIMANDO'])]))
        kpi3.metric("Pausados (RFI / Bloqueados)", len(df_activos[df_activos['Estado'].isin(['PENDIENTE CLIENTE'])]))
        kpi4.metric("SLA en Riesgo/Vencido", len(df_activos[df_activos['SLA_Status'].astype(str).str.contains('🟡|🔴')]))
        
        st.write("---")
        
        st.subheader("🔍 Filtros de Triage")
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        
        filtro_cliente = col_f1.selectbox("Filtrar por Cliente", ["Todos"] + list(df_activos['Cliente'].dropna().unique()))
        filtro_estado = col_f2.selectbox("Filtrar por Estado", ["Todos"] + list(df_activos['Estado'].dropna().unique()))
        filtro_consultor = col_f3.selectbox("Filtrar por Consultor", ["Todos"] + list(df_activos['Consultor'].dropna().unique()))
        filtro_sla = col_f4.selectbox("Filtrar por SLA Status", ["Todos"] + list(df_activos['SLA_Status'].dropna().unique()))
        
        df_filtrado = df_activos.copy()
        if filtro_cliente != "Todos": df_filtrado = df_filtrado[df_filtrado['Cliente'] == filtro_cliente]
        if filtro_estado != "Todos": df_filtrado = df_filtrado[df_filtrado['Estado'] == filtro_estado]
        if filtro_consultor != "Todos": df_filtrado = df_filtrado[df_filtrado['Consultor'] == filtro_consultor]
        if filtro_sla != "Todos": df_filtrado = df_filtrado[df_filtrado['SLA_Status'] == filtro_sla]
        
        st.dataframe(df_filtrado[['id_ticket', 'Cliente', 'Urgencia', 'Consultor', 'Estado', 'SLA_Status', 'Salud_Ppto']], use_container_width=True)
        
        st.write("---")
        
        st.subheader("🔎 Inspección de Comunicaciones por Ticket")
        if not df_filtrado.empty:
            ticket_inspect = st.selectbox("Seleccione un ticket filtrado para ver su bitácora completa:", df_filtrado['id_ticket'].tolist())
            
            query_drill = f"""
                SELECT fecha_envio as fecha, remitente, destinatario, mensaje, 'Interno' as tipo
                FROM comunicaciones_internas WHERE id_ticket = '{ticket_inspect}'
                UNION ALL
                SELECT fecha_creacion as fecha, aprobado_por as remitente, 'Cliente' as destinatario, cuerpo_mensaje as mensaje, 'Externo' as tipo
                FROM comunicaciones_cliente WHERE id_ticket = '{ticket_inspect}'
                ORDER BY fecha DESC
            """
            df_comms = extraer_tabla(query_drill)
            
            if not df_comms.empty:
                st.dataframe(df_comms, use_container_width=True)
            else:
                st.info("Este ticket no tiene comunicaciones registradas.")
        else:
            st.warning("No hay tickets que coincidan con los filtros aplicados.")
            
    else:
        st.info("No hay tickets en operación en este momento.")

# --- 4. ESPACIO DE TRABAJO CONSULTORES (WORKBENCH OPERATIVO) ---
elif menu == "💬 Espacio de Trabajo (Consultores)":
    st.title("Workbench Operativo del Consultor")
    
    df_cons = extraer_tabla("SELECT id_consultor, nombre FROM consultores")
    cons_sel = st.selectbox("Identifícate como Consultor:", df_cons['nombre'].tolist())
    id_cons_sel = df_cons.loc[df_cons['nombre'] == cons_sel, 'id_consultor'].values[0]

    # Query ampliado para traer la data de la bitácora y telemetría (Incluyendo cliente a través del JOIN)
    query_tkt = f"""
        SELECT 
            t.id_ticket, t.urgencia, t.descripcion_cliente, t.plan_accion_sugerido, 
            t.horas_estimadas, t.horas_consumidas_ticket, t.fecha_creacion, 
            t.estado_ticket, t.bitacora_tecnica, t.ordenes_transporte, 
            t.objetos_sap, t.archivos_adjuntos, c.nombre_empresa as nombre_cliente
        FROM tickets t
        LEFT JOIN clientes c ON t.id_cliente = c.id_cliente
        WHERE t.id_consultor_asignado = '{id_cons_sel}' AND t.estado_ticket IN ('EN PROCESO', 'ESTIMANDO')
    """
    df_tickets = extraer_tabla(query_tkt)
    
    if not df_tickets.empty:
        ticket_sel = st.selectbox("Selecciona tu ticket de trabajo activo:", df_tickets['id_ticket'].tolist())
        datos_ticket = df_tickets[df_tickets['id_ticket'] == ticket_sel].iloc[0]
        
        # 1. HUD DE TELEMETRÍA
        st.markdown("---")
        hud1, hud2, hud3 = st.columns(3)
        hud1.metric("🚦 Urgencia (SLA)", datos_ticket['urgencia'].split(" ")[0], calcular_estado_sla(datos_ticket['fecha_creacion'], datos_ticket['estado_ticket']))
        hud2.metric("⏱️ Horas Consumidas", f"{datos_ticket['horas_consumidas_ticket']:.1f} h", f"De {datos_ticket['horas_estimadas']:.1f} h Estimadas", delta_color="off")
        hud3.metric("📊 Salud Presupuesto", calcular_salud_presupuesto(datos_ticket['horas_estimadas'], datos_ticket['horas_consumidas_ticket']))
        st.markdown("---")
        
        # 2. SISTEMA DE PESTAÑAS DEL WORKBENCH
        tab_contexto, tab_interno, tab_cliente = st.tabs(["📋 Contexto y Bitácora SAP", "🏢 Chat Interno (Tracker IA)", "🤝 Chat con Cliente"])
        
        with tab_contexto:
            col_izq, col_der = st.columns(2)
            with col_izq:
                st.subheader("Historia de Usuario")
                st.info(f"{datos_ticket['descripcion_cliente']}")
                if datos_ticket.get('archivos_adjuntos'):
                    st.caption(f"📎 **Evidencias adjuntas:** {datos_ticket['archivos_adjuntos']}")
                
                st.success(f"**Plan de Acción RAG / Supervisor:**\n\n{datos_ticket['plan_accion_sugerido']}")
                
                # Botón de TL;DR (Resumen IA)
                if st.button("🧠 Generar Resumen IA del Historial (TL;DR)", use_container_width=True):
                    with st.spinner("Leyendo todo el historial de chats..."):
                        q_tldr = f"SELECT fecha_envio as fecha, remitente, mensaje FROM comunicaciones_internas WHERE id_ticket = '{ticket_sel}' UNION ALL SELECT fecha_creacion as fecha, aprobado_por as remitente, cuerpo_mensaje as mensaje FROM comunicaciones_cliente WHERE id_ticket = '{ticket_sel}' ORDER BY fecha ASC"
                        historial_completo = extraer_tabla(q_tldr).to_string()
                        prompt_tldr = f"Lee este historial de soporte:\n{historial_completo}\n\nEscribe estrictamente 3 viñetas cortas: 1) Problema raíz. 2) Acciones tomadas. 3) Qué falta por hacer. No agregues saludos."
                        resumen_ia, _, _ = llamar_agente_texto(prompt_tldr)
                        st.warning(f"**TL;DR:**\n\n{resumen_ia}")

            with col_der:
                st.subheader("Bitácora Técnica Estructurada")
                with st.form("bitacora_form"):
                    notas = st.text_area("Notas OSS / Resolución de la Falla:", value=datos_ticket['bitacora_tecnica'], height=120)
                    ots = st.text_input("Órdenes de Transporte (OTs):", value=datos_ticket['ordenes_transporte'])
                    objs = st.text_input("Objetos SAP Tocados:", value=datos_ticket['objetos_sap'])
                    
                    if st.form_submit_button("💾 Guardar Cambios en Bitácora", use_container_width=True):
                        ejecutar_sql("UPDATE tickets SET bitacora_tecnica=?, ordenes_transporte=?, objetos_sap=? WHERE id_ticket=?",
                                     (notas, ots, objs, ticket_sel))
                        st.success("Bitácora guardada exitosamente.")
                        time.sleep(1)
                        st.rerun()
                
                # Asistente de Cierre Técnico
                if st.button("🤖 Auto-Redactar Nota de Cierre", type="primary", use_container_width=True):
                    with st.spinner("Redactando cierre formal..."):
                        prompt_cierre = f"Redacta una nota formal de resolución para el cliente {datos_ticket['nombre_cliente']}. Problema: {datos_ticket['descripcion_cliente']}. Solución técnica aplicada: {datos_ticket['bitacora_tecnica']}. OTs: {datos_ticket['ordenes_transporte']}. Tono corporativo."
                        nota_cierre, _, _ = llamar_agente_texto(prompt_cierre)
                        st.info(f"**Borrador Sugerido (Cópialo para enviarlo):**\n\n{nota_cierre}")
        
        with tab_interno:
            chat_df = extraer_tabla(f"SELECT remitente, destinatario, mensaje, fecha_envio FROM comunicaciones_internas WHERE id_ticket = '{ticket_sel}' ORDER BY fecha_envio ASC")
            for index, row in chat_df.iterrows():
                is_user = row['remitente'] == cons_sel
                with st.chat_message("user" if is_user else "ai"):
                    st.markdown(f"**{row['remitente']}** a **{row['destinatario']}** ({row['fecha_envio'][:16]})\n\n{row['mensaje']}")

            archivos_chat_cons = st.file_uploader("📎 Anexar evidencia técnica (Logs, Capturas)", accept_multiple_files=True, key=f"up_cons_{ticket_sel}")
            st.caption("💡 *Nota: Después de enviar tu reporte, presiona la 'X' en el archivo adjunto para limpiar la bandeja.*")
            respuesta = st.chat_input("Escribe tu reporte de avance, horas consumidas o resolución...")
            
            if respuesta:
                nombres_archivos, contenido_ia_extra = procesar_adjuntos_multimodal(archivos_chat_cons, ticket_sel)
                mensaje_final = respuesta
                if nombres_archivos:
                    mensaje_final += f"\n\n📎 *Archivos anexos: {', '.join(nombres_archivos)}*"
                    
                ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                             (ticket_sel, cons_sel, "Tracker IA / Supervisor", mensaje_final, False))
                with st.spinner("Tracker IA está analizando tu reporte y evidencias..."):
                    prompt_base = f"El consultor {cons_sel} envió este reporte: '{mensaje_final}'. Analízalo y extrae la información en el JSON. Si falta detalle técnico, horas o estado final, marca extraccion_completa como false."
                    prompt_multimodal = [{"type": "text", "text": prompt_base}] + contenido_ia_extra
                    datos_track, t_in, t_out = llamar_agente(prompt_multimodal, EvaluacionTracker)
                    
                    if datos_track:
                        c_ia = calcular_costo_claude(t_in, t_out)
                        ejecutar_sql("UPDATE tickets SET tokens_usados = tokens_usados + ?, costo_ia = costo_ia + ? WHERE id_ticket = ?", (t_in + t_out, c_ia, ticket_sel))
                        
                        if datos_track.extraccion_completa:
                            ejecutar_sql("UPDATE tickets SET estado_ticket = ?, horas_consumidas_ticket = horas_consumidas_ticket + ? WHERE id_ticket = ?", (datos_track.estado_ticket, datos_track.horas_invertidas, ticket_sel))
                            ejecutar_sql("UPDATE clientes SET horas_consumidas = horas_consumidas + ? WHERE id_cliente = (SELECT id_cliente FROM tickets WHERE id_ticket = ?)", (datos_track.horas_invertidas, ticket_sel))
                            
                            if datos_track.estado_ticket == "RESUELTO":
                                ejecutar_sql("UPDATE consultores SET tickets_activos = MAX(0, tickets_activos - 1) WHERE id_consultor = ?", (id_cons_sel,))
                            
                            ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                                         (ticket_sel, "Tracker IA", cons_sel, f"Reporte procesado exitosamente. Tiempo debitado: {datos_track.horas_invertidas}h. Nuevo estado: {datos_track.estado_ticket}.", False))
                            ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                                         (ticket_sel, "Tracker IA", "Supervisor", f"El consultor {cons_sel} reportó un avance válido de {datos_track.horas_invertidas}h. Nuevo estado del ticket: {datos_track.estado_ticket}.", False))
                        else:
                            ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                                         (ticket_sel, "Tracker IA", cons_sel, datos_track.pregunta_seguimiento, True))
                st.rerun()
                
        with tab_cliente:
            chat_cli_df = extraer_tabla(f"SELECT aprobado_por as remitente, cuerpo_mensaje as mensaje, fecha_creacion FROM comunicaciones_cliente WHERE id_ticket = '{ticket_sel}' ORDER BY fecha_creacion ASC")
            for index, row in chat_cli_df.iterrows():
                is_internal = row['remitente'] != 'Cliente'
                with st.chat_message("assistant" if is_internal else "user"):
                    st.markdown(f"**{row['remitente']}** ({row['fecha_creacion'][:16]})\n\n{row['mensaje']}")
            
            archivos_chat_cons_cli = st.file_uploader("📎 Anexar documento o captura para el cliente (Opcional)", accept_multiple_files=True, key=f"up_cons_cli_{ticket_sel}")
            st.caption("💡 *Nota: Después de enviar tu mensaje, presiona la 'X' en el archivo adjunto para limpiar la bandeja.*")        
            resp_to_cli = st.chat_input("Pide información técnica o confirmaciones al cliente...")
            
            if resp_to_cli:
                nombres_archivos, _ = procesar_adjuntos_multimodal(archivos_chat_cons_cli, ticket_sel)
                mensaje_final = resp_to_cli
                if nombres_archivos:
                    mensaje_final += f"\n\n📎 *Archivos anexos: {', '.join(nombres_archivos)}*"
                    
                ejecutar_sql("INSERT INTO comunicaciones_cliente (id_ticket, estado_mensaje, cuerpo_mensaje, aprobado_por) VALUES (?, 'Enviado', ?, ?)", (ticket_sel, mensaje_final, cons_sel))
                st.rerun()
    else: st.info("No tienes tickets activos asignados (En Proceso / Estimando).")

# --- 5. TORRE DE CONTROL PMO (AGENTE 5) ---
elif menu == "📈 Torre de Control PMO (Gerencia)":
    st.title("Agente 5: PMO Advisor IA")
    
    st.subheader("💰 Finanzas, Costo IA y Satisfacción (CSAT)")
    
    df_tickets = extraer_tabla("SELECT * FROM tickets")
    df_clientes = extraer_tabla("SELECT id_cliente, nombre_empresa, horas_contratadas, horas_consumidas FROM clientes")
    
    if not df_tickets.empty and not df_clientes.empty:
        metricas = st.columns(4)
        metricas[0].metric("Total Tickets", len(df_tickets))
        metricas[1].metric("Horas Cobrables (Total)", f"{df_clientes['horas_consumidas'].sum():.1f}")
        metricas[2].metric("Inversión API IA ($)", f"${df_tickets['costo_ia'].sum():.3f}")
        csat_avg = df_tickets[df_tickets['calificacion_cliente'] > 0]['calificacion_cliente'].mean()
        metricas[3].metric("CSAT Promedio", f"{csat_avg:.1f} ⭐" if pd.notna(csat_avg) else "N/A")
        
        reporte_cli = pd.merge(df_clientes, df_tickets, on='id_cliente', how='left')
        resumen_cli = reporte_cli.groupby('nombre_empresa').agg(
            Tickets_Totales=('id_ticket', 'count'),
            Abiertos=('estado_ticket', lambda x: (x.isin(['NUEVO', 'EN PROCESO', 'ESTIMANDO', 'PENDIENTE CLIENTE'])).sum()),
            Resueltos=('estado_ticket', lambda x: (x == 'RESUELTO').sum()),
            Horas_Consumidas=('horas_consumidas', 'max'),
            Costo_API_IA=('costo_ia', 'sum'),
            CSAT_Promedio=('calificacion_cliente', lambda x: x[x > 0].mean() if len(x[x > 0]) > 0 else None)
        ).reset_index()
        
        st.dataframe(
            resumen_cli.style.format({
                "Horas_Consumidas": "{:.1f}",
                "Costo_API_IA": "${:.3f}",
                "CSAT_Promedio": lambda x: f"{x:.1f} ⭐" if pd.notnull(x) else "N/A"
            }),
            use_container_width=True
        )
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎯 Control de Portafolio (Burn Rate)")
        if not df_clientes.empty:
            df_cli_chart = df_clientes.copy()
            df_cli_chart.set_index('nombre_empresa', inplace=True)
            st.bar_chart(df_cli_chart[['horas_consumidas', 'horas_contratadas']])
            for index, row in df_clientes.iterrows():
                if row['horas_consumidas'] >= row['horas_contratadas']:
                    st.error(f"⚠️ {row['nombre_empresa']}: Riesgo de rentabilidad. Horas agotadas.")

    with col2:
        st.subheader("⚖️ Capacidad Operativa (WIP Limits)")
        df_cons = extraer_tabla("SELECT nombre, limite_wip, id_consultor FROM consultores")
        if not df_cons.empty:
            tickets_wip = extraer_tabla("SELECT id_consultor_asignado, count(id_ticket) as wip_real FROM tickets WHERE estado_ticket IN ('EN PROCESO', 'ESTIMANDO') GROUP BY id_consultor_asignado")
            df_cons_chart = pd.merge(df_cons, tickets_wip, left_on='id_consultor', right_on='id_consultor_asignado', how='left').fillna(0)
            
            df_plot = df_cons_chart[['nombre', 'wip_real', 'limite_wip']].copy()
            df_plot.set_index('nombre', inplace=True)
            st.bar_chart(df_plot)
            
            for index, row in df_cons_chart.iterrows():
                carga = (row['wip_real'] / row['limite_wip'] * 100) if row['limite_wip'] > 0 else 0
                if carga >= 85:
                    st.warning(f"⚠️ {row['nombre']}: Riesgo de Burnout (Carga {carga:.0f}%).")
            
    st.write("---")
    
    st.subheader("💬 Asesoría Estratégica Interactiva (PMO)")
    
    contexto_db = f"PORTAFOLIO:\n{df_clientes.to_json(orient='records')}\nCAPACIDAD:\n{df_cons.to_json(orient='records')}\nTICKETS Y FINANZAS:\n{df_tickets.to_json(orient='records')}"
    
    if st.button("🧠 Generar Informe Estratégico PMO", type="primary", use_container_width=True):
        with st.spinner("Agente 5 analizando la base de datos completa..."):
            prompt_pmo = f"Analiza los siguientes datos financieros y operativos: {contexto_db}. Redacta un informe ejecutivo evaluando la rentabilidad (costo IA vs Horas cobradas), satisfacción del cliente y riesgos del equipo."
            informe_pmo, t_in, t_out = llamar_agente_texto(prompt_pmo, rol="PMO")
            
            ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                         ("PMO-LOG", "Agente PMO (Agente 5)", "Gerencia", f"**INFORME EJECUTIVO PMO:**\n\n{informe_pmo}", False))
            st.rerun()

    chat_pmo_df = extraer_tabla("SELECT remitente, mensaje, fecha_envio FROM comunicaciones_internas WHERE id_ticket = 'PMO-LOG' ORDER BY fecha_envio ASC")
    
    for index, row in chat_pmo_df.iterrows():
        is_gerencia = row['remitente'] == 'Gerencia'
        with st.chat_message("user" if is_gerencia else "assistant"):
            st.markdown(f"**{row['remitente']}** ({row['fecha_envio'][:16]})\n\n{row['mensaje']}")
            
    pregunta_gerente = st.chat_input("Haz consultas sobre el ROI, costos IA o equipo...")
    if pregunta_gerente:
        ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                     ("PMO-LOG", "Gerencia", "Agente PMO (Agente 5)", pregunta_gerente, False))
        
        with st.spinner("Procesando telemetría..."):
            historial_reciente = ""
            if not chat_pmo_df.empty:
                ultimos_mensajes = chat_pmo_df.tail(5)
                historial_reciente = "\n".join([f"{row['remitente'].upper()}: {row['mensaje']}" for _, row in ultimos_mensajes.iterrows()])
            
            prompt_conversacional = f"CONTEXTO OPERATIVO Y FINANCIERO ACTUAL:\n{contexto_db}\n\nHISTORIAL DE LA REUNIÓN:\n{historial_reciente}\n\nNueva petición de la gerencia: {pregunta_gerente}\n\nResponde como el Agente 5 PMO Advisor."
            respuesta_ia, t_in, t_out = llamar_agente_texto(prompt_conversacional, rol="PMO")
            
            ejecutar_sql("INSERT INTO comunicaciones_internas (id_ticket, remitente, destinatario, mensaje, requiere_accion) VALUES (?, ?, ?, ?, ?)",
                         ("PMO-LOG", "Agente PMO (Agente 5)", "Gerencia", respuesta_ia, False))
        st.rerun()

# --- 6. CONFIGURACIÓN MAESTRA ---
elif menu == "⚙️ Configuración Maestra (Admin)":
    st.title("Administración de Datos Maestros")
    tab_cons, tab_cli, tab_rag = st.tabs(["👨‍💻 Consultores", "🏢 Clientes", "🧠 RAG Knowledge Base"])
    with tab_cons: st.dataframe(extraer_tabla("SELECT * FROM consultores"), use_container_width=True)
    with tab_cli: st.dataframe(extraer_tabla("SELECT * FROM clientes"), use_container_width=True)
    
    with tab_rag:
        st.subheader("Inyección Dinámica de Base de Conocimientos")
        
        col1, col2 = st.columns(2)
        with col1:
            prob_historico = st.text_area("Problema Histórico (Lo que reportó el usuario):", value="El usuario no puede ver la ganancia bruta en la factura SAP.")
        with col2:
            sol_historica = st.text_area("Solución Técnica (Lo que hizo el consultor):", value="Verificar autorizaciones. Habilitar V_VBRK_VKO en SU53.")
            
        if st.button("🧠 Vectorizar y Guardar en BD", type="primary"):
            with st.spinner("Conectando con Voyage AI..."):
                vector = obtener_embedding(prob_historico)
                if vector:
                    ejecutar_sql("CREATE TABLE IF NOT EXISTS BaseConocimiento (id INTEGER PRIMARY KEY AUTOINCREMENT, problema_historico TEXT, solucion TEXT, vector_embedding TEXT)")
                    ejecutar_sql("INSERT INTO BaseConocimiento (problema_historico, solucion, vector_embedding) VALUES (?, ?, ?)", (prob_historico, sol_historica, json.dumps(vector)))
                    st.success("✅ Conocimiento vectorizado e inyectado en SQLite exitosamente.")
        
        st.write("---")
        st.subheader("Base de Conocimiento Actual (Vectores Guardados)")
        df_rag = extraer_tabla("SELECT id, problema_historico, solucion FROM BaseConocimiento")
        if not df_rag.empty:
            st.dataframe(df_rag, use_container_width=True)
        else:
            st.info("La base de conocimiento RAG está vacía.")