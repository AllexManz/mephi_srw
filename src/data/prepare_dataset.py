import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import pandas as pd
from tqdm import tqdm
from dataset import SecurityDataset
import os
from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# Загружаем переменные окружения из .env файла
load_dotenv()

@hydra.main(version_base=None, config_path="../configs", config_name="base")
class DatasetPreparation:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        # Используем значения из конфигурации
        self.output_dir = Path(self.cfg.dataset.dataset.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CVE конфигурация
        self.nvd_api_key = self.cfg.env.NVD_API_KEY
        self.cve_days_lookback = self.cfg.dataset.cve.days_lookback
        self.cve_results_per_page = self.cfg.dataset.cve.results_per_page
        self.cve_max_results = self.cfg.dataset.cve.max_results
        self.cve_min_cvss_score = self.cfg.dataset.cve.min_cvss_score
        
        # MITRE ATT&CK конфигурация
        self.mitre_attack_url = self.cfg.env.MITRE_ATTACK_URL
        self.mitre_include_tactics = self.cfg.dataset.mitre_attack.include_tactics
        self.mitre_include_techniques = self.cfg.dataset.mitre_attack.include_techniques
        
        # Security Documentation конфигурация
        self.include_security_docs = self.cfg.dataset.security_docs.include
        self.max_docs_examples = self.cfg.dataset.security_docs.max_examples
        
    def download_mitre_attack(self) -> List[Dict[str, Any]]:
        """Загрузка и обработка данных MITRE ATT&CK."""
        print("Downloading MITRE ATT&CK data...")
        
        if not (self.mitre_include_tactics or self.mitre_include_techniques):
            print("MITRE ATT&CK data download disabled in configuration")
            return []
        
        # Используем URL из конфигурации
        enterprise_url = self.mitre_attack_url
        headers = {
            "Accept": "application/json",
            "User-Agent": "Security-LLM-Dataset-Preparation"
        }
        
        try:
            response = requests.get(enterprise_url, headers=headers)
            response.raise_for_status()
            enterprise_data = response.json()
            
            examples = []
            
            # Обработка тактик
            if self.mitre_include_tactics:
                for tactic in enterprise_data.get("objects", []):
                    if tactic.get("type") == "x-mitre-tactic":
                        # Получаем краткое имя тактики из external_references
                        tactic_id = next(
                            (ref.get("external_id") for ref in tactic.get("external_references", [])
                             if ref.get("source_name") == "mitre-attack"),
                            "Unknown"
                        )
                        
                        examples.append({
                            "instruction": "Опиши тактику атаки и предоставь рекомендации по защите",
                            "input": f"Тактика: {tactic.get('name')}\nID: {tactic_id}\nОписание: {tactic.get('description', '')}",
                            "output": f"Тактика '{tactic.get('name')}' (ID: {tactic_id}) относится к категории {tactic.get('x_mitre_shortname', '')}. "
                                     f"Эта тактика описывает {tactic.get('description', '')}\n\n"
                                     f"Рекомендации по защите:\n"
                                     f"1. Регулярно проводить аудит безопасности\n"
                                     f"2. Внедрить систему обнаружения вторжений\n"
                                     f"3. Обеспечить мониторинг сетевой активности\n"
                                     f"4. Поддерживать актуальность систем безопасности"
                        })
            
            # Обработка техник
            if self.mitre_include_techniques:
                for technique in enterprise_data.get("objects", []):
                    if technique.get("type") == "attack-pattern":
                        # Получаем ID техники из external_references
                        technique_id = next(
                            (ref.get("external_id") for ref in technique.get("external_references", [])
                             if ref.get("source_name") == "mitre-attack"),
                            "Unknown"
                        )
                        
                        examples.append({
                            "instruction": "Опиши технику атаки и предоставь рекомендации по защите",
                            "input": f"Техника: {technique.get('name')}\nID: {technique_id}\nОписание: {technique.get('description', '')}",
                            "output": f"Техника '{technique.get('name')}' (ID: {technique_id}) "
                                     f"описывает {technique.get('description', '')}\n\n"
                                     f"Рекомендации по защите:\n"
                                     f"1. Внедрить систему обнаружения аномалий\n"
                                     f"2. Регулярно обновлять системы безопасности\n"
                                     f"3. Проводить обучение сотрудников\n"
                                     f"4. Внедрить принцип наименьших привилегий"
                        })
            
            print(f"Successfully downloaded {len(examples)} examples from MITRE ATT&CK")
            return examples
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading MITRE ATT&CK data: {str(e)}")
            print("Falling back to local examples...")
            return self._get_local_mitre_examples()
    
    def _get_local_mitre_examples(self) -> List[Dict[str, Any]]:
        """Возвращает локальные примеры MITRE ATT&CK."""
        return [
            {
                "instruction": "Опиши тактику атаки и предоставь рекомендации по защите",
                "input": "Тактика: Initial Access\nОписание: The adversary is trying to get into your network.",
                "output": "Тактика Initial Access описывает методы, которые используют злоумышленники для получения первоначального доступа к сети.\n\n"
                         "Рекомендации по защите:\n"
                         "1. Внедрить систему обнаружения вторжений\n"
                         "2. Обеспечить фильтрацию входящего трафика\n"
                         "3. Регулярно обновлять системы безопасности\n"
                         "4. Проводить обучение сотрудников по вопросам безопасности"
            },
            {
                "instruction": "Опиши технику атаки и предоставь рекомендации по защите",
                "input": "Техника: Phishing\nОписание: Adversaries may send phishing messages to gain access to victim systems.",
                "output": "Техника Phishing описывает методы, при которых злоумышленники отправляют фишинговые сообщения для получения доступа к системам жертв.\n\n"
                         "Рекомендации по защите:\n"
                         "1. Внедрить систему фильтрации email\n"
                         "2. Проводить регулярное обучение сотрудников\n"
                         "3. Использовать многофакторную аутентификацию\n"
                         "4. Внедрить систему обнаружения фишинговых атак"
            }
        ]
    
    def download_cve_data(self) -> List[Dict[str, Any]]:
        """Загрузка и обработка данных CVE."""
        print("Downloading CVE data...")
        
        if not self.nvd_api_key:
            print("Warning: NVD API key not provided. Using local examples instead.")
            return self._get_local_cve_examples()
        
        # Получаем вчерашнюю дату в UTC как конечную точку
        current_time = datetime.now(timezone.utc)
        end_date = current_time - timedelta(days=1)  # Используем вчерашний день
        
        # Используем 118 дней как период (немного меньше максимального для надежности)
        max_days = min(self.cve_days_lookback, 118)
        start_date = end_date - timedelta(days=max_days)
        
        # Форматируем даты в ISO 8601 формат, который ожидает NVD API
        start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
        
        print(f"Current UTC time: {current_time}")
        print(f"Using end date: {end_date}")
        print(f"Requesting CVE data from {start_date_str} to {end_date_str}")
        print(f"Note: Using {max_days} days lookback period (limited to 118 days for reliability)")
        
        url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
        
        headers = {
            "apiKey": self.nvd_api_key,
            "Accept": "application/json"
        }
        
        examples = []
        total_results = 0
        start_index = 0
        
        while total_results < self.cve_max_results:
            params = {
                "pubStartDate": start_date_str,
                "pubEndDate": end_date_str,
                "resultsPerPage": self.cve_results_per_page,
                "startIndex": start_index
            }
            
            try:
                print(f"Requesting CVE data (page {start_index // self.cve_results_per_page + 1})...")
                response = requests.get(url, headers=headers, params=params)
                
                if response.status_code != 200:
                    print(f"Error response from NVD API:")
                    print(f"Status code: {response.status_code}")
                    print(f"Response headers: {dict(response.headers)}")
                    print(f"Response body: {response.text[:500]}...")
                    response.raise_for_status()
                
                data = response.json()
                
                if "vulnerabilities" not in data:
                    print(f"Unexpected API response structure: {data.keys()}")
                    break
                
                # Проверяем, есть ли еще результаты
                if not data.get("vulnerabilities"):
                    print("No more CVE results available")
                    break
                
                for vuln in data.get("vulnerabilities", []):
                    if total_results >= self.cve_max_results:
                        print(f"Reached maximum number of CVE results ({self.cve_max_results})")
                        break
                    
                    cve = vuln.get("cve", {})
                    
                    # Получаем описание на английском языке
                    description = next(
                        (desc.get("value") for desc in cve.get("descriptions", [])
                         if desc.get("lang") == "en"),
                        "No description available"
                    )
                    
                    # Получаем CVSS score
                    cvss_score = 0.0
                    if "metrics" in cve:
                        if "cvssMetricV31" in cve["metrics"]:
                            cvss_score = float(cve["metrics"]["cvssMetricV31"][0]["cvssData"]["baseScore"])
                        elif "cvssMetricV2" in cve["metrics"]:
                            cvss_score = float(cve["metrics"]["cvssMetricV2"][0]["cvssData"]["baseScore"])
                    
                    # Пропускаем уязвимости с низким CVSS score
                    if cvss_score < self.cve_min_cvss_score:
                        continue
                    
                    # Получаем вектор атаки
                    attack_vector = "Unknown"
                    if "metrics" in cve and "cvssMetricV31" in cve["metrics"]:
                        attack_vector = cve["metrics"]["cvssMetricV31"][0]["cvssData"].get("attackVector", "Unknown")
                    
                    # Получаем дату публикации
                    published = cve.get("published", "Unknown")
                    
                    examples.append({
                        "instruction": "Проанализируй уязвимость и предоставь рекомендации по исправлению",
                        "input": f"CVE ID: {cve.get('id')}\n"
                                f"Дата публикации: {published}\n"
                                f"Описание: {description}\n"
                                f"CVSS Score: {cvss_score}\n"
                                f"Вектор атаки: {attack_vector}",
                        "output": f"Уязвимость {cve.get('id')} имеет следующие характеристики:\n\n"
                                 f"Дата публикации: {published}\n"
                                 f"Описание: {description}\n\n"
                                 f"Уровень опасности: {cvss_score} (по шкале CVSS)\n"
                                 f"Вектор атаки: {attack_vector}\n\n"
                                 f"Рекомендации по исправлению:\n"
                                 f"1. Обновить уязвимое ПО до последней версии\n"
                                 f"2. Применить соответствующие патчи безопасности\n"
                                 f"3. Проверить систему на наличие других уязвимостей\n"
                                 f"4. Внедрить механизмы мониторинга для обнаружения попыток эксплуатации\n"
                                 f"5. Провести аудит безопасности системы"
                    })
                    total_results += 1
                
                # Проверяем, есть ли еще страницы
                if len(data.get("vulnerabilities", [])) < self.cve_results_per_page:
                    print("No more pages available")
                    break
                
                start_index += self.cve_results_per_page
                
            except requests.exceptions.RequestException as e:
                print(f"Error downloading CVE data: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    print(f"Error details: {e.response.text[:500]}...")
                print("Falling back to local examples...")
                return self._get_local_cve_examples()
        
        print(f"Successfully downloaded {len(examples)} CVE examples (filtered by CVSS score >= {self.cve_min_cvss_score})")
        return examples
    
    def _get_local_cve_examples(self) -> List[Dict[str, Any]]:
        """Возвращает локальные примеры CVE."""
        return [
            {
                "instruction": "Проанализируй уязвимость и предоставь рекомендации по исправлению",
                "input": "CVE ID: CVE-2023-1234\n"
                        "Описание: A critical vulnerability in Apache Log4j allows remote code execution through JNDI lookup.\n"
                        "CVSS Score: 9.8",
                "output": "Уязвимость CVE-2023-1234 (Log4Shell) имеет следующие характеристики:\n\n"
                         "Описание: Критическая уязвимость в Apache Log4j, позволяющая выполнение удаленного кода через JNDI lookup.\n\n"
                         "Рекомендации по исправлению:\n"
                         "1. Обновить Apache Log4j до версии 2.17.0 или выше\n"
                         "2. Применить патчи безопасности\n"
                         "3. Проверить все системы на наличие уязвимой версии\n"
                         "4. Внедрить WAF для блокировки попыток эксплуатации"
            },
            {
                "instruction": "Проанализируй уязвимость и предоставь рекомендации по исправлению",
                "input": "CVE ID: CVE-2023-5678\n"
                        "Описание: Buffer overflow in OpenSSL allows remote attackers to execute arbitrary code.\n"
                        "CVSS Score: 8.5",
                "output": "Уязвимость CVE-2023-5678 имеет следующие характеристики:\n\n"
                         "Описание: Переполнение буфера в OpenSSL, позволяющее удаленным атакующим выполнять произвольный код.\n\n"
                         "Рекомендации по исправлению:\n"
                         "1. Обновить OpenSSL до последней версии\n"
                         "2. Проверить все сервисы, использующие OpenSSL\n"
                         "3. Внедрить ASLR и другие механизмы защиты от переполнения буфера\n"
                         "4. Настроить мониторинг попыток эксплуатации"
            }
        ]
    
    def download_nist_framework(self) -> List[Dict[str, Any]]:
        """Загрузка данных из NIST Cybersecurity Framework."""
        print("Downloading NIST Cybersecurity Framework data...")
        
        url = "https://raw.githubusercontent.com/usnistgov/csf/main/framework.json"
        headers = {
            "Accept": "application/json",
            "User-Agent": "Security-LLM-Dataset-Preparation"
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            examples = []
            for category in data.get("framework", {}).get("categories", []):
                examples.append({
                    "instruction": "Опиши категорию кибербезопасности и предоставь рекомендации по внедрению",
                    "input": f"Категория: {category.get('name')}\n"
                            f"Описание: {category.get('description', '')}",
                    "output": f"Категория '{category.get('name')}' в NIST Cybersecurity Framework:\n\n"
                             f"Описание: {category.get('description', '')}\n\n"
                             f"Рекомендации по внедрению:\n"
                             f"1. Провести оценку текущего состояния\n"
                             f"2. Разработать план внедрения\n"
                             f"3. Внедрить необходимые меры контроля\n"
                             f"4. Настроить мониторинг и аудит\n"
                             f"5. Регулярно проводить оценку эффективности"
                })
            
            print(f"Successfully downloaded {len(examples)} examples from NIST Framework")
            return examples
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading NIST Framework data: {str(e)}")
            return []

    def download_mitre_cwe(self) -> List[Dict[str, Any]]:
        """Загрузка данных из MITRE CWE."""
        print("Downloading MITRE CWE data...")
        
        url = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
        headers = {
            "Accept": "application/zip",
            "User-Agent": "Security-LLM-Dataset-Preparation"
        }
        
        try:
            # Скачиваем и распаковываем XML
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            import zipfile
            import io
            import xml.etree.ElementTree as ET
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                xml_content = zip_ref.read(zip_ref.namelist()[0])
            
            root = ET.fromstring(xml_content)
            examples = []
            
            for weakness in root.findall(".//{http://cwe.mitre.org/cwe-6}Weakness"):
                cwe_id = weakness.get("ID", "Unknown")
                name = weakness.find(".//{http://cwe.mitre.org/cwe-6}Title").text
                description = weakness.find(".//{http://cwe.mitre.org/cwe-6}Description").text
                
                examples.append({
                    "instruction": "Опиши уязвимость и предоставь рекомендации по защите",
                    "input": f"CWE ID: {cwe_id}\n"
                            f"Название: {name}\n"
                            f"Описание: {description}",
                    "output": f"Уязвимость {name} (CWE-{cwe_id}):\n\n"
                             f"Описание: {description}\n\n"
                             f"Рекомендации по защите:\n"
                             f"1. Провести анализ кода на наличие уязвимости\n"
                             f"2. Внедрить соответствующие меры защиты\n"
                             f"3. Обновить документацию по безопасности\n"
                             f"4. Провести обучение разработчиков\n"
                             f"5. Регулярно проводить тестирование на наличие уязвимости"
                })
            
            print(f"Successfully downloaded {len(examples)} examples from MITRE CWE")
            return examples
            
        except (requests.exceptions.RequestException, zipfile.BadZipFile, ET.ParseError) as e:
            print(f"Error downloading MITRE CWE data: {str(e)}")
            return []

    def download_owasp_data(self) -> List[Dict[str, Any]]:
        """Загрузка данных из OWASP (Top 10 и Cheat Sheets)."""
        print("Downloading OWASP data...")
        
        # OWASP Top 10
        top10_url = "https://raw.githubusercontent.com/OWASP/Top10/master/2021/data.json"
        # OWASP Cheat Sheets
        cheat_sheets_url = "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/Index.md"
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "Security-LLM-Dataset-Preparation"
        }
        
        examples = []
        
        try:
            # Загрузка Top 10
            response = requests.get(top10_url, headers=headers)
            response.raise_for_status()
            top10_data = response.json()
            
            for risk in top10_data.get("data", []):
                examples.append({
                    "instruction": "Опиши риск безопасности и предоставь рекомендации по защите",
                    "input": f"Риск: {risk.get('name')}\n"
                            f"Описание: {risk.get('description', '')}\n"
                            f"Примеры: {risk.get('example', '')}",
                    "output": f"Риск безопасности '{risk.get('name')}':\n\n"
                             f"Описание: {risk.get('description', '')}\n\n"
                             f"Примеры: {risk.get('example', '')}\n\n"
                             f"Рекомендации по защите:\n"
                             f"1. Провести оценку уязвимости\n"
                             f"2. Внедрить соответствующие меры защиты\n"
                             f"3. Обновить документацию\n"
                             f"4. Провести обучение персонала\n"
                             f"5. Регулярно проводить тестирование"
                })
            
            # Загрузка Cheat Sheets
            response = requests.get(cheat_sheets_url)
            response.raise_for_status()
            cheat_sheets_content = response.text
            
            # Парсим Markdown для получения списка cheat sheets
            import re
            cheat_sheet_links = re.findall(r'\[(.*?)\]\((.*?)\)', cheat_sheets_content)
            
            for title, url in cheat_sheet_links:
                if "CheatSheet" in title:
                    try:
                        sheet_response = requests.get(url)
                        sheet_response.raise_for_status()
                        content = sheet_response.text
                        
                        # Извлекаем основное содержание
                        main_content = re.search(r'##.*?\n(.*?)(?=##|$)', content, re.DOTALL)
                        if main_content:
                            examples.append({
                                "instruction": "Опиши лучшие практики безопасности и предоставь рекомендации",
                                "input": f"Cheat Sheet: {title}\n"
                                        f"Содержание: {main_content.group(1)[:500]}...",
                                "output": f"Лучшие практики безопасности из {title}:\n\n"
                                         f"{main_content.group(1)}\n\n"
                                         f"Рекомендации по внедрению:\n"
                                         f"1. Оценить текущее состояние\n"
                                         f"2. Разработать план внедрения\n"
                                         f"3. Внедрить необходимые меры\n"
                                         f"4. Провести обучение\n"
                                         f"5. Регулярно обновлять практики"
                            })
                    except requests.exceptions.RequestException:
                        continue
            
            print(f"Successfully downloaded {len(examples)} examples from OWASP")
            return examples
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading OWASP data: {str(e)}")
            return []

    def prepare_security_documentation(self) -> List[Dict[str, Any]]:
        """Подготовка примеров из различных источников документации по безопасности."""
        if not self.include_security_docs:
            print("Security documentation examples disabled in configuration")
            return []
            
        print("Preparing security documentation examples...")
        
        all_examples = []
        
        # Загружаем данные из различных источников
        all_examples.extend(self.download_nist_framework())
        all_examples.extend(self.download_mitre_cwe())
        all_examples.extend(self.download_owasp_data())
        
        # Добавляем базовые примеры, если не удалось загрузить данные
        if not all_examples:
            print("Falling back to local examples...")
            all_examples.extend(self._get_local_security_examples())
        
        # Ограничиваем количество примеров
        if len(all_examples) > self.max_docs_examples:
            print(f"Limiting security documentation examples to {self.max_docs_examples} (from {len(all_examples)})")
            all_examples = all_examples[:self.max_docs_examples]
        
        print(f"Prepared {len(all_examples)} security documentation examples")
        return all_examples

    def _get_local_security_examples(self) -> List[Dict[str, Any]]:
        """Возвращает локальные примеры безопасности, если не удалось загрузить данные."""
        return [
            {
                "instruction": "Опиши основные принципы безопасности и предоставь рекомендации по их внедрению",
                "input": "Основные принципы информационной безопасности: конфиденциальность, целостность, доступность",
                "output": "Основные принципы информационной безопасности (CIA):\n\n"
                         "1. Конфиденциальность (Confidentiality):\n"
                         "   - Защита информации от несанкционированного доступа\n"
                         "   - Использование шифрования\n"
                         "   - Внедрение систем контроля доступа\n\n"
                         "2. Целостность (Integrity):\n"
                         "   - Обеспечение точности и полноты данных\n"
                         "   - Использование контрольных сумм\n"
                         "   - Внедрение систем обнаружения изменений\n\n"
                         "3. Доступность (Availability):\n"
                         "   - Обеспечение доступа к информации авторизованным пользователям\n"
                         "   - Внедрение систем резервного копирования\n"
                         "   - Обеспечение отказоустойчивости систем"
            }
        ]
    
    def prepare_dataset(self) -> None:
        """Подготовка полного датасета."""
        # Собираем примеры из всех источников
        all_examples = []
        
        if self.mitre_include_tactics or self.mitre_include_techniques:
            all_examples.extend(self.download_mitre_attack())
        
        all_examples.extend(self.download_cve_data())
        
        if self.include_security_docs:
            all_examples.extend(self.prepare_security_documentation())
        
        import random
        random.shuffle(all_examples)
        
        # Разделяем на train и validation
        split_idx = int(len(all_examples) * self.cfg.dataset.dataset.train_val_split_ratio)
        train_examples = all_examples[:split_idx]
        val_examples = all_examples[split_idx:]
        
        with open(self.output_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump(train_examples, f, ensure_ascii=False, indent=2)
        
        with open(self.output_dir / "eval.json", "w", encoding="utf-8") as f:
            json.dump(val_examples, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset prepared and saved to {self.output_dir}")
        print(f"Total examples: {len(all_examples)}")
        print(f"Train examples: {len(train_examples)}")
        print(f"Validation examples: {len(val_examples)}")

@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig):
    """Main function to prepare the dataset."""
    print("Loading configuration...")
    print(OmegaConf.to_yaml(cfg))
    
    # Create dataset preparation instance
    dataset_prep = DatasetPreparation(cfg)
    
    # Prepare dataset
    dataset_prep.prepare_dataset()
    
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main() 